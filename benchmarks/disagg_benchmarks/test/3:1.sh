#!/bin/bash

# Requirement: 2x GPUs.


# Model: /root/models/Qwen/Qwen2.5-7B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex

export HF_ENDPOINT=https://hf-mirror.com
export PATH="/root/miniconda3/bin:$PATH"
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate test

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

export VLLM_USE_V1=0
launch_chunked_prefill() {
  model="/root/models/Qwen/Qwen2.5-7B-Instruct"
  # disagg prefill
  CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.85 &

  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.85 &

  CUDA_VISIBLE_DEVICES=2 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8300 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.85 &

CUDA_VISIBLE_DEVICES=3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8400 \
    --max-model-len 10000 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.85 &

  wait_for_server 8100
  wait_for_server 8200
  wait_for_server 8300
  wait_for_server 8400
  python3 round_robin_proxy.py &
  sleep 1
}
# --max_num_seqs 256 \

launch_disagg_prefill() {
  model="/root/models/Qwen/Qwen2.5-7B-Instruct" 
  # disagg prefill
  CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config \
    '{"kv_connector":"NcclStoreConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

  CUDA_VISIBLE_DEVICES=1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config \
    '{"kv_connector":"NcclStoreConnector","kv_role":"kv_producer","kv_rank":1,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

    CUDA_VISIBLE_DEVICES=2 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8300 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config \
    '{"kv_connector":"NcclStoreConnector","kv_role":"kv_producer","kv_rank":2,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

    CUDA_VISIBLE_DEVICES=3 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8400 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config \
    '{"kv_connector":"NcclStoreConnector","kv_role":"kv_consumer","kv_rank":3,"kv_parallel_size":4,"kv_buffer_size":4e9}' &

    # CUDA_VISIBLE_DEVICES=4 python3 \
    # -m vllm.entrypoints.openai.api_server \
    # --model $model \
    # --port 8200 \
    # --max-model-len 10000 \
    # --gpu-memory-utilization 0.85 \
    # --kv-transfer-config \
    # '{"kv_connector":"NcclStoreConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":4e9}' &

    # CUDA_VISIBLE_DEVICES=5 python3 \
    # -m vllm.entrypoints.openai.api_server \
    # --model $model \
    # --port 8200 \
    # --max-model-len 10000 \
    # --gpu-memory-utilization 0.85 \
    # --kv-transfer-config \
    # '{"kv_connector":"NcclStoreConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":4e9}' &

    # CUDA_VISIBLE_DEVICES=6 python3 \
    # -m vllm.entrypoints.openai.api_server \
    # --model $model \
    # --port 8200 \
    # --max-model-len 10000 \
    # --gpu-memory-utilization 0.85 \
    # --kv-transfer-config \
    # '{"kv_connector":"NcclStoreConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":4e9}' &

    # CUDA_VISIBLE_DEVICES=7 python3 \
    # -m vllm.entrypoints.openai.api_server \
    # --model $model \
    # --port 8200 \
    # --max-model-len 10000 \
    # --gpu-memory-utilization 0.85 \
    # --kv-transfer-config \
    # '{"kv_connector":"NcclStoreConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":4e9}' &
    

  wait_for_server 8100
  wait_for_server 8200
  wait_for_server 8300
  wait_for_server 8400
  python3 xpyd.py & # NcclStoreConnector PyNcclConnector
  sleep 1
}


benchmark() {
  results_folder="./results3_1"
  model="/root/models/Qwen/Qwen2.5-7B-Instruct"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=300
  qps=$1
  prefix_len=50
  input_len=$3
  output_len=$2
  tag=$4

  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len "$output_len" \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
          --request-rate "$qps" \
          --random-range-ratio 0.2 \
          --random-input-len 512 
  sleep 2
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results3_1
  mkdir results3_1

  default_output_len=20

  default_input_len=500

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  kill_gpu_processes   # 

  launch_disagg_prefill #1:1 12 10 8 6       2:1 20 18 16 14   3:1 26 24 22 20 , 4:1 38 34 30 26  5:1  50 56 42 38    
  for qps in 26 24 22 20 ; do
  benchmark $qps $default_output_len $default_input_len disagg_prefill
  done

  kill_gpu_processes

  launch_chunked_prefill
  for qps in 26 24 22 20 ; do
  benchmark $qps $default_output_len $default_input_len chunked_prefill
  done

  kill_gpu_processes

  python3 visualize_benchmark_results.py

}


main "$@"
