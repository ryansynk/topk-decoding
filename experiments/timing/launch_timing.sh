# USAGE: sbatch launch_timing.sh <decoding_strategy> [<k>]

# Ensure at least one argument (decode_strategy) is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: bash launch_timing.sh <decoding_strategy> [<k>]"
    exit 1
fi

decode_strategy=$1
k=$2

# Loop over N values
#for N in 4096 8192 16384 32768 65536 131072
for N in 4096 8192
  do
    if [[ "$decode_strategy" == "topk_full" || "$decode_strategy" == "topk_ivf" || "$decode_strategy" == "streaming_llm" ]]; then
        if [ -z "$k" ]; then
            echo "Error: <k> parameter is required for decoding_strategy '$decode_strategy'"
            exit 1
        fi
        python timing.py --N=${N} --decode_strategy="$decode_strategy" --k="$k" --device="rtxa6000"
    else
        python timing.py --N=${N} --decode_strategy="$decode_strategy" --device="rtxa6000"
    fi
  done
