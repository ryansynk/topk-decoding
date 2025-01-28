import pytest
from topk_decoding import topk_decoding as t

def test_import_convert_cache_to_topk():
    with pytest.raises(NotImplementedError) as e_info:
        t.convert_cache_to_topk(None)

def test_import_convert_model_to_topk():
    with pytest.raises(NotImplementedError) as e_info:
        t.convert_model_to_topk(None)
