#!/usr/bin/env bash
# run at project root dir
# Usage:
# bash baselines/mxml/scripts/train.sh zh video_sub resnet_i3d ANY_OTHER_PYTHON_ARGS
# use --eval_tasks_at_training ["VR", "SVMR", "VCMR"] --stop_task ["VR", "SVMR", "VCMR"] for
# use --lw_neg_q 0 --lw_neg_ctx 0 for training SVMR/SVMR only
# use --lw_st_ed 0 for training with VR only
dset_name=tvr  # see case below
ctx_mode=video_sub  # [video, sub, tef, video_sub, video_tef, sub_tef, video_sub_tef]
vid_feat_type=resnet_i3d  # [resnet, i3d, resnet_i3d]
feature_root=./mtvr_features
results_root=baselines/mxml/results
vid_feat_size=2048
bsz=128  # since we have two languages, real batch size will be 128 * 2
exp_id=exp
extra_args=()

if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
    if [[ ${dset_name} != "tvr" ]]; then
        echo "The use of subtitles is only supported in tvr."
        exit 1
    fi
fi


case ${dset_name} in
    tvr)
        train_path=data/tvr_zh_en_train_release.jsonl
        video_duration_idx_path=data/tvr_video2dur_idx.json
        if [[ ${vid_feat_type} == "i3d" ]]; then
            echo "Using I3D feature with shape 1024"
            vid_feat_path=${feature_root}/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5
            vid_feat_size=1024
        elif [[ ${vid_feat_type} == "resnet" ]]; then
            echo "Using ResNet feature with shape 2048"
            vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_cl-1.5.h5
            vid_feat_size=2048
        elif [[ ${vid_feat_type} == "resnet_i3d" ]]; then
            echo "Using concatenated ResNet and I3D feature with shape 2048+1024"
            vid_feat_path=${feature_root}/video_feature/tvr_resnet152_rgb_max_i3d_rgb600_avg_cat_cl-1.5.h5
            vid_feat_size=3072
            extra_args+=(--no_norm_vfeat)  # since they are already normalized.
        fi
        eval_split_name=val
        nms_thd=-1
        extra_args+=(--eval_path)
        extra_args+=(data/tvr_zh_en_val_release.jsonl)
        clip_length=1.5
        extra_args+=(--max_ctx_l)
        extra_args+=(100)  # max_ctx_l = 100 for clip_length = 1.5, only ~109/21825 has more than 100.
        extra_args+=(--max_pred_l)
        extra_args+=(16)
        desc_bert_path_en=${feature_root}/bert_feature_en/sub_query/tvr_query_pretrained_w_sub_query.h5
        desc_bert_path_zh=${feature_root}/bert_feature_zh/sub_query/tvr_query_pretrained_w_sub_query.h5
        for lang in en zh
        do
          if [[ ${ctx_mode} == *"sub"* ]] || [[ ${ctx_mode} == "sub" ]]; then
              echo "Running with sub."
              sub_bert_path=${feature_root}/bert_feature_${lang}/sub_query/tvr_sub_pretrained_w_sub_query_max_cl-1.5.h5
              sub_feat_size=768
              extra_args+=(--sub_feat_size)
              extra_args+=(${sub_feat_size})
              extra_args+=(--sub_bert_path_${lang})
              extra_args+=(${sub_bert_path})
          fi
        done
        ;;
    *)
        echo -n "Unknown argument"
        ;;
esac

echo "Start training with dataset [${dset_name}] in Context Mode [${ctx_mode}]"
echo "Extra args ${extra_args[@]}"
lw_neighbors=1.; neighbors_names=(query sub1 sub2)
PYTHONPATH=$PYTHONPATH:. python baselines/mxml/train_shared.py \
--dset_name=${dset_name} \
--eval_split_name=${eval_split_name} \
--nms_thd=${nms_thd} \
--results_root=${results_root} \
--train_path=${train_path} \
--desc_bert_path_en=${desc_bert_path_en} \
--desc_bert_path_zh=${desc_bert_path_zh} \
--video_duration_idx_path=${video_duration_idx_path} \
--vid_feat_path=${vid_feat_path} \
--clip_length=${clip_length} \
--vid_feat_size=${vid_feat_size} \
--ctx_mode=${ctx_mode} \
--exp_id=${exp_id} \
--bsz ${bsz} \
--no_cross_att \
--share_vid_enc \
--share_sub_enc \
--share_q_enc \
--lw_neighbors ${lw_neighbors} \
--neighbors_names ${neighbors_names[@]} \
${extra_args[@]} \
${@:1}

