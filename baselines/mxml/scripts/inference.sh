#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/mxml/scripts/inference.sh ANY_OTHER_PYTHON_ARGS
model_dir=$1
eval_split_name=$2
eval_path=standalone_eval/data_archive/tvr_zh_en_${eval_split_name}_archive.jsonl
tasks=()
tasks+=(VCMR)
tasks+=(SVMR)
tasks+=(VR)
echo "tasks ${tasks[@]}"
python baselines/mxml/inference_shared.py \
--model_dir ${model_dir} \
--tasks ${tasks[@]} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}
