#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/mxml/scripts/eval.sh ANY_OTHER_PYTHON_ARGS
eval_split_name=$1
submission_path=$2
save_path=$3
gt_path=standalone_eval/data_archive/tvr_zh_en_${eval_split_name}_archive.jsonl

python standalone_eval/eval.py \
--gt_path ${gt_path} \
--submission_path ${submission_path} \
--save_path ${save_path} \
${@:4}
