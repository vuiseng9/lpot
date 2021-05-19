#!/usr/bin/env bash
export LOGLEVEL=DEBUG
WORKDIR=/home/vchua/may17-lpot/lpot/examples/pytorch/huggingface_models/examples/text-classification

cd ${WORKDIR}
mkdir -p lpot-bert-base-cased-finetuned-mrpc_run

# CUDA_VISIBLE_DEVICES=0 1 
nohup python run_glue_tune.py \
     --tuned_checkpoint lpot-bert-base-cased-finetuned-mrpc_run \
     --model_name_or_path  ~/bert-zoo/bert-base-cased-finetuned-mrpc  \
     --task_name  mrpc  \
     --do_eval  \
     --max_seq_length  128 \
     --per_gpu_eval_batch_size  16 \
     --no_cuda  \
     --output_dir  lpot-bert-base-cased-finetuned-mrpc_run/hf-output-dir \
     --tune 2>&1 | tee lpot-bert-base-cased-finetuned-mrpc_run/log.lpot-bert-base-cased-finetuned-mrpc &
