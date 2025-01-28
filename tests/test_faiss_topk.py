import unittest
import math
import gc
import os
from shutil import rmtree
import torch
import torch.nn.functional as F
import einops
from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicFaissCache
from transformers.models.llama.modeling_llama import repeat_kv_db
from transformers.topk_attn import (
    naive_topk_attn,
    create_sparse_matrix,
    get_topk_via_faiss,
    get_topk_via_knn,
    topk_attn,
)
import numpy as np
import sys
import wandb
import time
import copy


class TestFaiss(unittest.TestCase):
    @staticmethod
    def create_config(k):
        config = PretrainedConfig()
        config.topk = k
        config.save_indices = False
        config.save_contexts = False
        config.save_attn_last = False
        config.save_attn_agg = False
        config.save_hist = False
        config.scratchpad = {"counter": 0}
        return config

    @staticmethod
    def create_pythia_model(k):
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-14m",
            torch_dtype=torch.bfloat16,
            attn_implementation="topk",
            topk=k,
            save_contexts=False,
            device_map="cuda",
        )
        return model

    
    @staticmethod
    def create_llama_model(name="meta-llama/Llama-2-7b-chat-hf", k=10, attn_implementation="topk"):
        model = (
            AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                save_contexts=False,
                low_cpu_mem_usage=True,
            )
            .cuda()
            .eval()
        )
        return model
    
    
    @staticmethod
    def create_llama_model_naive(k):
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            attn_implementation="naive_topk",
            topk=k,
            save_contexts=False,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )
        return model
    
    @staticmethod
    def create_small_model(name="./test_data/small_model", attn_implementation="topk"):
        small_topk_model = (
            AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                save_contexts=False,
                num_hidden_layers=1,
                hidden_size=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                ignore_mismatched_sizes=True,
            )
            .cuda()
            .eval()
        )
        return small_topk_model


    def assertAllclose(self, tensor1, tensor2, rtol=1e-5, atol=1e-8):
        diff = torch.abs(tensor1 - tensor2)
        max_diff = diff.max().item()
        allclose = diff > (atol + rtol * torch.abs(tensor2))
        num_diff = allclose.sum().item()
        diff_sums = allclose.sum(axis=-1)
        if num_diff > 0:
            self.fail(
                """Logits don't match. Total differences: {0}, max diff:  
                {1}, num diffs in last dimension: {2} out of shape 
                {3}""".format(num_diff, max_diff, diff_sums, tensor1.shape)
            )

    # TODO test with multiple values of k?
    def setUp(self):
        self.B = 1
        self.H = 32
        self.N = 512
        self.D = 128
        self.kv_heads = 4
        self.kv_groups = self.H // self.kv_heads

        self.k = 10
        torch.manual_seed(42)

        self.Q = torch.rand((self.B, self.H, self.N, self.D))
        self.K = torch.rand((self.B, self.kv_heads, self.N, self.D))
        self.V = torch.rand((self.B, self.H, self.N, self.D))
        K_repeat = einops.repeat(self.K, "B kv_heads N D -> B (kv_heads kv_groups) N D", kv_groups=self.kv_groups)
        self.attn_scores = self.Q @ K_repeat.mT / math.sqrt(self.D)
        self.mask = torch.triu(
            torch.full_like(self.attn_scores, fill_value=float("-inf")), diagonal=1
        )
        self.config = TestFaiss.create_config(self.k)

        self.model_str = "meta-llama/Llama-2-7b-chat-hf"
        # self.model_str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


    ################################################################################################
    # Tests Overview:
    #
    # - Atomic unit tests:
    #   1. test_create_sparse_matrix()                 - Tests that sparse matrix creation is equivalent to dense matrix creation
    #   2. test_create_sparse_matrix_with_mask()       - Same as above with fused causal
    #   3. test_get_topk_via_faiss_small()             - Tests that Faiss method extracts the correct top-k
    #   4. test_get_topk_via_faiss_large()             - Same as above but with large tensors to quantifiably show stochasticity issues with Faiss
    #   -  test_get_topk_via_knn_small()               - Deprecated, not currently used.
    #   -  test_get_topk_via_knn_large()               - Deprecated, not currently used.
    #   5. test_cache_serialization()                  - Tests that saving and loading cache to disk has no corruption
    #
    # - Intermediate Integration tests:
    #   6. test_topk_attn()                            - Tests that attention-layer outputs during construct mode match naive topk_attn() outputs
    #   7. test_construct_cache()                      - Tests that full-model outputs during construct mode have same logits as a naive topk_attn() call
    #   8. test_construct_cache_loop()                 - Same as above but loops over input in chunks of a fixed size
    #
    # - Full Integration tests:
    #   9. test_topk_generate_small_model()            - Tests that prefix + suffix combination generates identical output to normal attention. Tests integration with huggingface generate() API.
    #   10. test_topk_generate_full_model()            - Same as above but with Llama2-7b model.
    #   11. test_topk_generate_full_model()            - Same as above but loops over input in chunks of a fixed size
    #
    ################################################################################################

    # Atomic unit tests
    # 1.
    def test_create_sparse_matrix(self):
        """
        Tests the correctness of our method for constructing a sparse matrix from topk indices and values.
        Faiss is not used here, we just retrieve indices and values from torch.topk and use our routine to construct the sparse matrix.
        Our naive topk_attn() function that has already been evaluated to our satisfction returns a dense matrix (with -inf instead of zeros
        because it will be softmaxed later). If our sparse matrix creation routine is correct, we can turn it into a dense matrix and it should
        match the output of the dense matrix from naive topk_attn().
        """
        B, H, N, N = self.attn_scores.shape
        values, indices = torch.topk(self.attn_scores, self.config.topk, dim=-1)
        values = einops.rearrange(values, "B H N K -> (B H) N K")
        indices = einops.rearrange(indices, "B H N K -> (B H) N K")
        sparse_matrix = create_sparse_matrix(values, indices, N, mask=False) # Explicitly create without a causal mask.
        self.assertTrue(
            sparse_matrix.layout == torch.sparse_coo,
            f"Expected sparse format but got: {sparse_matrix.layout}",
        )

        # Confirm the dense topk matrix is equivalent to the sparse version we created
        dense_matrix = naive_topk_attn(self.config, self.attn_scores, self.Q, self.K, self.V)
        # Convert -inf to 0
        dense_matrix[dense_matrix == float("-inf")] = 0
        dense_matrix = einops.rearrange(dense_matrix, "B H N1 N2 -> (B H) N1 N2")
        
        self.assertTrue(
            dense_matrix.shape == sparse_matrix.to_dense().shape,
            f"Shape mismatch: {dense_matrix.shape} and {sparse_matrix.to_dense().shape}",
        )
        self.assertAllclose(dense_matrix, sparse_matrix.to_dense())
        self.assertTrue(torch.equal(dense_matrix, sparse_matrix.to_dense()))

    # 2.
    def test_create_sparse_matrix_with_mask(self):
        """
        Same as above, but we can save the n^2 memory cost of a triangular causal mask but fusing it into the sparse matrix creation routine.
        Test that this remains correct.
        """
        B, H, N, N = self.attn_scores.shape
        values, indices = torch.topk(self.attn_scores, self.config.topk, dim=-1)
        values = einops.rearrange(values, "B H N K -> (B H) N K")
        indices = einops.rearrange(indices, "B H N K -> (B H) N K")
        sparse_matrix = create_sparse_matrix(values, indices, N)  # mask=True by default
        self.assertTrue(
            sparse_matrix.layout == torch.sparse_coo,
            f"Expected sparse format but got: {sparse_matrix.layout}",
        )

        # Confirm the dense topk matrix is equivalent to the sparse version we created
        dense_matrix = naive_topk_attn(self.config, self.attn_scores, self.Q, self.K, self.V)
        dense_matrix = dense_matrix + self.mask
        # Convert -inf to 0
        dense_matrix[dense_matrix == float("-inf")] = 0
        dense_matrix = einops.rearrange(dense_matrix, "B H N1 N2 -> (B H) N1 N2")

        self.assertTrue(
            dense_matrix.shape == sparse_matrix.to_dense().shape,
            f"Shape mismatch: {dense_matrix.shape} and {sparse_matrix.to_dense().shape}",
        )
        self.assertAllclose(dense_matrix, sparse_matrix.to_dense())
        self.assertTrue(torch.equal(dense_matrix, sparse_matrix.to_dense()))

    # 3.
    def test_get_topk_via_faiss_small(self):
        """
        Tests correctness of our use of Faiss index retrieval for getting the topk scores from an attention matrix.
        Small tensor we expect to have no tie-break problems.
        """
        B, H, N, D = 1, 12, 10, 8
        kv_heads = 3
        kv_groups = H // kv_heads
        Q = torch.rand((B, H, N, D))
        K = torch.rand((B, kv_heads, N, D))
        k = 2
        torch_indices, faiss_indices, torch_values, faiss_values = self.topk_tester(Q, K, D, k, kv_heads, kv_groups, get_topk_via_faiss)
        
        self.assertTrue(
            torch_values.shape == faiss_values.to_dense().shape,
            f"Shape mismatch: {torch_values.shape} and {faiss_values.to_dense().shape}",
        )
        self.assertAllclose(torch_values, faiss_values, rtol=1e-5, atol=1e-8)
        self.assertTrue(
            torch.equal(torch_indices, faiss_indices),
            "We expect the indices to be the same 99% of the time, so for this small case we expect them to be the same."
        )
        # Note: We expect the values to be the same up to a 1e-5 tolerance, but expect about 3% to be off in the 1e-6 range.
        # See "_large" test below for testing against that case.

    # 4.
    def test_get_topk_via_faiss_large(self):
        """
        Tests correctness of our use of Faiss index retrieval for getting the topk scores from an attention matrix.
        Large case where we expect tiebreak problems.
        """
        B, H, N, D = 100, 32, 100, 128
        kv_heads = 4
        kv_groups = H // kv_heads
        Q = torch.rand((B, H, N, D))
        K = torch.rand((B, kv_heads, N, D))
        k = 10
        torch_indices, faiss_indices, torch_values, faiss_values = self.topk_tester(Q, K, D, k, kv_heads, kv_groups, get_topk_via_faiss)
        
        self.assertTrue(
            torch_values.shape == faiss_values.to_dense().shape,
            f"Shape mismatch: {torch_values.shape} and {faiss_values.to_dense().shape}",
        )
        self.assertAllclose(torch_values, faiss_values, rtol=1e-5, atol=1e-8)

        self.assertFalse(
            torch.equal(torch_indices, faiss_indices),
            "We expect the indices to be the same 99% of the time, and this large case presents enough chances that we expect tiebreak problems to appear."
        )
        # Note: Value correctness asserted in topk_tester()

    def topk_tester(self, Q, K, D, k, kv_heads, kv_groups, topk_function):
        """
        Tests our topk-extraction routine by comparing a given topk function with torch.topk().
        Used to test both get_topk_via_faiss() and get_topk_via_knn().
        As noted above, we expect that the value will not be bit-accurate (off at 1e-6), but the indices might be off
        entirely, because of the tiebreaking process.
        """
        # Torch-based topk
        K_repeat = einops.repeat(K, "B kv_heads N D -> B (kv_heads kv_groups) N D", kv_groups=kv_groups)
        attn_scores = Q @ K_repeat.mT / math.sqrt(D)
        values, indices = torch.topk(attn_scores, k, dim=-1)
        values = einops.rearrange(values, "B H N K -> (B H) N K")
        indices = einops.rearrange(indices, "B H N K -> (B H) N K")

        # Target topk to test
        Q = einops.rearrange(Q, "B H N D -> (B H) N D")
        K = einops.rearrange(K, "B H N D -> (B H) N D")
        if topk_function == get_topk_via_faiss:
            key_database = DynamicFaissCache.create_key_database(K)
            key_database = repeat_kv_db(key_database, kv_groups)
            faiss_values, faiss_indices = topk_function(k, Q, key_database, kv_heads, kv_groups)
        # elif topk_function == get_topk_via_knn:
        #     faiss_values, faiss_indices = topk_function(k, Q, K)
        else:
            raise ValueError(f"Expected either get_topk_via_faiss but got {topk_function}")

        return indices, faiss_indices, values, faiss_values
    

    # def test_get_topk_via_knn_small(self):
    #     """
    #     Tests correctness of our use of Faiss GPU knn for getting the topk scores from an attention matrix.
    #     Small tensor we expect to have no tie-break problems.
    #     """
    #     B, H, N, D = 1, 1, 10, 8
    #     Q = torch.rand((B, H, N, D))
    #     K = torch.rand((B, H, N, D))
    #     k = 2
    #     torch_indices, faiss_indices = self.topk_tester(Q, K, D, k, get_topk_via_knn)
    #     self.assertTrue(
    #         torch.equal(torch_indices, faiss_indices),
    #         "We expect the indices to be the same 99% of the time, so for this small case we expect them to be the same."
    #     )
    #     # Note: Value correctness asserted in topk_tester()


    # def test_get_topk_via_knn_large(self):
    #     """
    #     Tests correctness of our use of Faiss GPU knn for getting the topk scores from an attention matrix.
    #     Large case where we expect tiebreak problems.
    #     """
    #     B, H, N, D = 100, 32, 100, 128
    #     Q = torch.rand((B, H, N, D))
    #     K = torch.rand((B, H, N, D))
    #     k = 10
    #     torch_indices, faiss_indices = self.topk_tester(Q, K, D, k, get_topk_via_knn)
    #     self.assertFalse(
    #         torch.equal(torch_indices, faiss_indices),
    #         "We expect the indices to be the same 99% of the time, and this large case presents enough chances that we expect tiebreak problems to appear."
    #     )
    #     # Note: Value correctness asserted in topk_tester()

    # 5.
    def test_cache_serialization(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        prompt = (
            "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
            "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
            "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
            "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
            "he stumbled upon Tessa slowly making her way across the path. "
            "What happened next?"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        topk_k = 30
        prefix_inputs = inputs.input_ids[:, :topk_k].cuda()
        suffix_inputs = inputs.input_ids[:, topk_k:].cuda()
        prompt_length = inputs["input_ids"].shape[1]
        suffix_length = suffix_inputs.shape[1]


        topk_model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="topk")
        with torch.no_grad():
            cache = topk_model.construct_cache(prefix_inputs.cuda(), k_construct=topk_k)
            DynamicFaissCache.save_cache(cache, "test_data/test_cache_dir")
            
            # Delete the cache and model to ensure we are properly testing a load from disk
            topk_model.cpu()
            del topk_model
            del cache
            gc.collect()
            torch.cuda.empty_cache()
            
            topk_model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="topk")
            loaded_cache = DynamicFaissCache.load_cache(
                "test_data/test_cache_dir", 
                topk_model.config.num_hidden_layers, 
                topk_model.config.num_key_value_heads
            )

            topk_outputs = topk_model.generate(
                suffix_inputs.cuda(),
                max_new_tokens=80,
                use_cache=True,
                past_key_values=loaded_cache,
                topk_k=topk_k,
                query_mode=True,
                do_sample=False,
                num_beams=1,
            )
            topk_output_ids = topk_outputs[0][suffix_length:]

        topk_model.cpu()
        del topk_model
        torch.cuda.empty_cache()
        gc.collect()

        model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="eager")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(), max_new_tokens=80, do_sample=False, num_beams=1
            )
            output_ids = outputs[0][prompt_length:]

        rmtree("./test_data/test_cache_dir")
        topk_output_str = tokenizer.decode(topk_output_ids)
        output_str = tokenizer.decode(output_ids)
        debug_str = "\nEXPECTED OUTPUT:\n" + output_str + "\nTOPK OUTPUT:\n" + topk_output_str
        self.assertTrue(torch.equal(output_ids, topk_output_ids), msg=debug_str)

        model.cpu()            
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Intermediate Integration tests
    # 6.
    def test_topk_attn(self):
        """
        Tests function topk_attn()
        Compare xhat (hidden_states) output from  attention layer for naive_topk_attn() vs. topk_attn().
        topk_construction ouput does three things:
        1. Gets topk indices and values via Faiss (as opposed to torch.topk())
        2. Constructs a sparse matrix from the topk indices and values
        3. Does a sparse matrix multiplication to get the hidden states
        
        Note: the outputs of topk value selection (1) are not bit-accurate. They differ at the 1e-6 level.
        Also, the indices *might* differ if there are to values that are very close together (imagine faiss.topk and torch.topk
        having different tie-breaking rules). Since there are those nuances in (1), the output of naive_topk_attn() and topk_attn()
        *will* be different if given large enough vectors to operate on. However, we expect this unit test to pass with small inputs, demonstrating
        that the two functions only differe in these nuances, and not in the core functionality.
        """
        K_repeat = einops.repeat(self.K, "B kv_heads N D -> B (kv_heads kv_groups) N D", kv_groups=self.kv_groups)
        score = naive_topk_attn(self.config, self.attn_scores, self.Q, K_repeat, self.V)
        score = score + self.mask
        attn = F.softmax(score, dim=-1)
        attn = torch.nan_to_num(attn)
        xhat = attn @ self.V
        key_db = DynamicFaissCache.create_key_database(self.K)
        key_db = repeat_kv_db(key_db, self.kv_groups)
        suffix_keys = torch.empty(0)
        prefix_keys = torch.empty(0)
        xhat_faiss = topk_attn(
            self.config.topk, self.Q, suffix_keys, prefix_keys, key_db, self.V, self.B, self.H, self.kv_heads, self.kv_groups, construct_mode=True
        ).cpu()
        
        self.assertTrue(
                xhat.shape == xhat_faiss.shape,
                f"Shape mismatch: {xhat.shape} and {xhat_faiss.shape}",
            )
        self.assertAllclose(xhat, xhat_faiss)
        self.assertFalse(torch.equal(xhat, xhat_faiss), "We expect torch-based topk and faiss-based topk to differ in the 1e-6 range, but not in the 1e-5 range")

    # 7.
    def test_construct_cache(self):
        """
        Tests function model.construct_cache()
        This compares the logits of a forward pass using naive topk with forward pass using faiss topk, and where the
        keys and values are cached along the way.

        Note: This faces the same nuances as above, but with model the size of Llama-2-7b, we now have large vectors and enough of them
        that we
        """
        if torch.cuda.is_available():
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

            tokenizer = AutoTokenizer.from_pretrained(self.model_str)
            prompt = (
                "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
                "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
                "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
                "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
                "he stumbled upon Tessa slowly making her way across the path. "
                "What happened next?"
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            topk_k = inputs.input_ids.shape[-1]

            topk_model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="topk")
            with torch.no_grad():
                cache, logits_faiss = topk_model.construct_cache(inputs.input_ids.cuda(), k_construct=topk_k, output_logits=True)

            # Delete topk_model object to make room for new model
            topk_model.cpu()
            del topk_model
            gc.collect()
            torch.cuda.empty_cache()

            model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="eager")
            with torch.no_grad():
                logits = model(inputs.input_ids.cuda())["logits"]

            self.assertTrue(
                logits.shape == logits_faiss.shape,
                f"Shape mismatch: {logits.shape} and {logits_faiss.shape}",
            )
            pred_logits = logits[0, -1, :]
            pred_logits_faiss = logits_faiss[0, -1, :]
            # Get the top 10 predicted token IDs
            v, i = pred_logits.topk(10)
            v_faiss, i_faiss = pred_logits_faiss.topk(10)
            naive_str = tokenizer.batch_decode(i)
            faiss_str = tokenizer.batch_decode(i_faiss)
            self.assertTrue(
                len(set(naive_str).intersection(set(faiss_str))) >= 9, 
                f"At least 9/10 top tokens should overlap, instead got {naive_str} and {faiss_str}")
            
            model.cpu()            
            del model
            gc.collect()
            torch.cuda.empty_cache()

        else:
            self.assertTrue(False, "CUDA Unavailable")


    
    # Full Integration tests
    # 9.
    def test_topk_generate_small_model(self):
        """Tests the output of generate with a small llama model

        Using randomly generated weights, tests that a 1-layer llama model with small
        hidden dimensions generates the same output when using normal attention and
        topk attention
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        prompt = (
            "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
            "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
            "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
            "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
            "he stumbled upon Tessa slowly making her way across the path. "
            "What happened next?"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        topk_k = 30
        prefix_inputs = inputs.input_ids[:, :topk_k].cuda()
        suffix_inputs = inputs.input_ids[:, topk_k:].cuda()
        prompt_length = inputs["input_ids"].shape[1]
        suffix_length = suffix_inputs.shape[1]

        # We have a config for a small model in "tests/topk/test_data/small_model". Wrap this in a
        # try-except to handle calling these tests from either root or the tests directory and to
        # provide a helpful error message if the model is not found.
        try:
            location = "tests/topk/test_data/small_model"
            small_model = TestFaiss.create_small_model(name=location, attn_implementation="eager")
            small_topk_model = TestFaiss.create_small_model(name=location, attn_implementation="topk")
        except OSError as e:
            try:
                location = "test_data/small_model"
                small_model = TestFaiss.create_small_model(name=location, attn_implementation="eager")
                small_topk_model = TestFaiss.create_small_model(name=location, attn_implementation="topk")
            except OSError as e:
                message = f"Couldn't find a model at {location}. Please make sure the model exists and you are calling from the root of that location."
                raise OSError(message) from e


        with torch.no_grad():
            cache = small_topk_model.construct_cache(prefix_inputs.cuda(), k_construct=topk_k)
            topk_outputs = small_topk_model.generate(
                suffix_inputs.cuda(),
                max_new_tokens=10,
                use_cache=True,
                past_key_values=cache,
                topk_k=topk_k,
                query_mode=True,
                do_sample=False,
                num_beams=1,
            )
            topk_output_ids = topk_outputs[0][suffix_length:]

            outputs = small_model.generate(
                inputs.input_ids.cuda(), max_new_tokens=10, do_sample=False, num_beams=1
            )
            output_ids = outputs[0][prompt_length:]

        topk_output_str = tokenizer.decode(topk_output_ids)
        output_str = tokenizer.decode(output_ids)
        debug_str = "\nEXPECTED OUTPUT:\n" + output_str + "\nTOPK OUTPUT:\n" + topk_output_str
        self.assertTrue(torch.equal(output_ids, topk_output_ids), msg=debug_str)

        small_model.cpu()
        del small_model
        gc.collect()
        small_topk_model.cpu()
        del small_topk_model
        gc.collect()
        torch.cuda.empty_cache()

    # 10.
    def test_topk_generate_full_model(self):
        """Tests that generation with topk and generation with normal attention match.

        Test that models using 'topk' attention implementation generate the same tokens
        as those using standard attention when using the full value of k.

        This test sets up a deterministic environment for reproducibility by configuring
        the PyTorch cuBLAS workspace. It then loads a tokenizer and two variants of the
        same model: one with 'topk' attention and the other with 'eager' attention.

        The process involves:
        1. Tokenizing a prompt and dividing it into prefix and suffix inputs.
        2. Initializing both model variants.
        3. Constructing a cache for the 'topk' model with the prefix inputs.
        4. Generating tokens with both models using the respective configurations.
        5. Comparing the generated token sequences from both models to ensure they are identical.

        Assertions:
        - The token sequences generated by the 'topk' model and the 'eager' model must be equal.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        prompt = (
            "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
            "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
            "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
            "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
            "he stumbled upon Tessa slowly making her way across the path. "
            "What happened next?"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        topk_k = 30
        prefix_inputs = inputs.input_ids[:, :topk_k].cuda()
        suffix_inputs = inputs.input_ids[:, topk_k:].cuda()
        prompt_length = inputs["input_ids"].shape[1]
        suffix_length = suffix_inputs.shape[1]
        topk_k = prefix_inputs.shape[-1]

        topk_model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="topk")
        with torch.no_grad():
            cache = topk_model.construct_cache(prefix_inputs.cuda(), k_construct=topk_k)
            topk_outputs = topk_model.generate(
                suffix_inputs.cuda(),
                max_new_tokens=80,
                use_cache=True,
                past_key_values=cache,
                topk_k=topk_k,
                query_mode=True,
                do_sample=False,
                num_beams=1,
            )
            topk_output_ids = topk_outputs[0][suffix_length:]

        topk_model.cpu()
        del topk_model
        gc.collect()
        torch.cuda.empty_cache()

        model = TestFaiss.create_llama_model(name=self.model_str, k=topk_k, attn_implementation="eager")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(), max_new_tokens=80, do_sample=False, num_beams=1
            )
            output_ids = outputs[0][prompt_length:]

        topk_output_str = tokenizer.decode(topk_output_ids)
        output_str = tokenizer.decode(output_ids)
        debug_str = "\nEXPECTED OUTPUT:\n" + output_str + "\nTOPK OUTPUT:\n" + topk_output_str
        self.assertTrue(torch.equal(output_ids, topk_output_ids), msg=debug_str)

        model.cpu()            
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def test_unrolled_generate_full_model(self):
        """Tests that generation with topk and generation with normal attention match.

        Test that models using 'topk' attention implementation generate the same tokens
        as those using standard attention when using the full value of k.

        This test sets up a deterministic environment for reproducibility by configuring
        the PyTorch cuBLAS workspace. It then loads a tokenizer and two variants of the
        same model: one with 'topk' attention and the other with 'eager' attention.

        The process involves:
        1. Tokenizing a prompt and dividing it into prefix and suffix inputs.
        2. Initializing both model variants.
        3. Constructing a cache for the 'topk' model with the prefix inputs.
        4. Generating tokens with both models using the respective configurations.
        5. Comparing the generated token sequences from both models to ensure they are identical.

        Assertions:
        - The token sequences generated by the 'topk' model and the 'eager' model must be equal.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        prompt = (
            "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
            "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
            "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
            "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
            "he stumbled upon Tessa slowly making her way across the path. "
            "What happened next?"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        unrolled_model = TestFaiss.create_llama_model(name=self.model_str, attn_implementation="unrolled")
        with torch.no_grad():
            unrolled_outputs = unrolled_model.generate(
                inputs.input_ids.cuda(), max_new_tokens=80, do_sample=False, num_beams=1, use_cache=False
            )
            unrolled_output_ids = unrolled_outputs[0]

        unrolled_model.cpu()
        del unrolled_model
        gc.collect()
        torch.cuda.empty_cache()

        model = TestFaiss.create_llama_model(name=self.model_str, attn_implementation="eager")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(), max_new_tokens=80, do_sample=False, num_beams=1, use_cache=False
            )
            output_ids = outputs[0]

        unrolled_output_str = tokenizer.decode(unrolled_output_ids)
        output_str = tokenizer.decode(output_ids)
        debug_str = "\nEXPECTED OUTPUT:\n" + output_str + "\nUNROLLED OUTPUT:\n" + unrolled_output_str
        self.assertTrue(torch.equal(output_ids, unrolled_output_ids), msg=debug_str)

        model.cpu()            
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()

