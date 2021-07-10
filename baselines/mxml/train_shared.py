import os
import time
import json
import pprint
import copy
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from baselines.mxml.config_shared import BaseOptions
from baselines.mxml.model_xml_shared import XMLShared
from baselines.xml.start_end_dataset import start_end_collate, prepare_batch_inputs
from baselines.mxml.dataset_shared import DatasetShared, start_end_collate_shared
from baselines.xml.inference import eval_epoch
from baselines.mxml.inference_shared import start_inference
from baselines.xml.optimization import BertAdam
from baselines.xml.train import set_seed, rm_key_from_odict, get_datasets
from utils.basic_utils import AverageMeter
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def train_epoch(model, train_loaders, optimizer, opt, epoch_i, training=True):
    logger.info("use train_epoch func for training: {}".format(training))
    languages = ["en", "zh"]
    # use_paired_datasets = opt.lw_neighbors > 0
    use_paired_datasets = opt.use_paired_datasets
    model.train(mode=training)
    if opt.hard_negtiave_start_epoch != -1:
        if epoch_i >= opt.hard_negtiave_start_epoch:
            model.set_hard_negative(True, opt.hard_pool_size)
        if epoch_i >= opt.harder_negtiave_start_epoch:
            model.set_hard_negative(True, opt.harder_pool_size)
        if epoch_i >= opt.hardest_negtiave_start_epoch:
            model.set_hard_negative(True, opt.hardest_pool_size)
    if opt.train_span_start_epoch != -1 and epoch_i >= opt.train_span_start_epoch:
        model.set_train_st_ed(opt.lw_st_ed)

    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    _loss_meters = OrderedDict(loss_st_ed=AverageMeter(),
                               loss_vr=AverageMeter(),
                               loss_overall=AverageMeter())
    loss_meters = {}
    for lang in languages:
        loss_meters[lang] = copy.deepcopy(_loss_meters)
    if use_paired_datasets:
        # only add to en to avoid redundancy
        loss_meters["en"]["lang_align"] = AverageMeter()

    # single_epoch_total_steps = len(train_loaders["en"])
    single_epoch_total_steps = len(train_loaders) \
        if use_paired_datasets else len(train_loaders["en"])
    timer_dataloading = time.time()
    enumerator = enumerate(train_loaders) \
        if use_paired_datasets else enumerate(zip(train_loaders["en"], train_loaders["zh"]))
    tqdm_iterator = tqdm(
        enumerator,
        desc="Training Iteration",
        total=single_epoch_total_steps
    )
    for batch_idx, (batch_en, batch_zh) in tqdm_iterator:
        global_step = epoch_i * single_epoch_total_steps + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)

        # prepare inputs
        batch = dict(en=batch_en, zh=batch_zh)
        timer_start = time.time()
        model_inputs = {}
        for lang in languages:
            model_inputs[lang] = prepare_batch_inputs(
                batch[lang][1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)

        # forward, twice, one for each language
        timer_start = time.time()
        loss, loss_dicts = model(model_inputs)
        model_forward_time.update(time.time() - timer_start)

        # backward, only once.
        timer_start = time.time()
        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)

            opt.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            for lang, loss_dict in loss_dicts.items():
                for k, v in loss_dict.items():
                    opt.writer.add_scalar("Train/{}-{}".format(lang, k), v, global_step)

        for lang, loss_dict in loss_dicts.items():
            for k, v in loss_dict.items():
                loss_meters[lang][k].update(float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    if training:
        for lang in languages:
            to_write = opt.train_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters[lang].items()]))
            with open(opt.train_log_filepath + lang, "a") as f:
                f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} "
              "min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n"
              "".format(dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
                        model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        for lang in languages:
            for k, v in loss_meters[lang].items():
                opt.writer.add_scalar("Eval_Loss/{}-{}".format(lang, k), v.avg, epoch_i)


def train(model, opt, datasets):
    languages = ["en", "zh"]
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    # use_paired_datasets = opt.lw_neighbors > 0
    use_paired_datasets = opt.use_paired_datasets
    train_loader_cfg = dict(
        dataset=None,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )
    if use_paired_datasets:
        train_loader_cfg["dataset"] = DatasetShared(
            en_dataset=datasets["train"]["en"], zh_dataset=datasets["train"]["zh"])
        train_loader_cfg["collate_fn"] = start_end_collate_shared
        train_loaders = DataLoader(**train_loader_cfg)
    else:
        train_loaders = {}
        for lang in languages:
            train_loader_cfg["dataset"] = datasets["train"][lang]
            train_loaders[lang] = DataLoader(**train_loader_cfg)

    train_eval_loader_cfg = dict(
        dataset=None,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    if use_paired_datasets:
        train_eval_loader_cfg["dataset"] = DatasetShared(
            en_dataset=datasets["train_eval"]["en"], zh_dataset=datasets["train_eval"]["zh"])
        train_eval_loader_cfg["collate_fn"] = start_end_collate_shared
        train_eval_loaders = DataLoader(**train_eval_loader_cfg)
    else:
        train_eval_loaders = {}
        for lang in languages:
            train_eval_loader_cfg["dataset"] = datasets["train_eval"][lang]
            train_eval_loaders[lang] = DataLoader(**train_eval_loader_cfg)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    # single_epoch_total_steps = len(train_loaders["en"])
    single_epoch_total_steps = len(train_loaders) \
        if use_paired_datasets else len(train_loaders["en"])
    total_optimization_steps = single_epoch_total_steps * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         weight_decay=opt.wd,
                         warmup=opt.lr_warmup_proportion,
                         t_total=total_optimization_steps,
                         schedule="warmup_linear")

    prev_best_score = 0.
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0
    eval_tasks_at_training = opt.eval_tasks_at_training  # VR is computed along with VCMR
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            # with torch.autograd.detect_anomaly():
            train_epoch(model, train_loaders, optimizer, opt, epoch_i, training=True)
        global_step = (epoch_i + 1) * single_epoch_total_steps
        if opt.eval_path is not None:
            with torch.no_grad():
                train_epoch(model, train_eval_loaders, optimizer, opt, epoch_i, training=False)

                results = {}
                for lang in languages:
                    save_submission_filename = \
                        f"{lang}_latest_{opt.dset_name}_{opt.eval_split_name}" \
                        f"_predictions_{'_'.join(eval_tasks_at_training)}.json"
                    _model = getattr(model, f"xml_{lang}")
                    metrics_no_nms, metrics_nms, latest_file_paths = \
                        eval_epoch(_model, datasets["eval"][lang], opt, save_submission_filename,
                                   tasks=eval_tasks_at_training, max_after_nms=100)
                    results[lang] = dict(metrics_no_nms=metrics_no_nms,
                                         metrics_nms=metrics_nms,
                                         latest_file_paths=latest_file_paths)

                    # log
                    to_write = opt.eval_log_txt_formatter.format(
                        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                        epoch=epoch_i,
                        eval_metrics_str=json.dumps(metrics_no_nms))
                    with open(opt.eval_log_filepath + lang, "a") as f:
                        f.write(to_write)
                    logger.info("{} metrics_no_nms {}".format(
                        lang, pprint.pformat(rm_key_from_odict(metrics_no_nms, rm_suffix="by_type"), indent=4)))
                    logger.info("{} metrics_nms {}".format(lang, pprint.pformat(metrics_nms, indent=4)))

                    # metrics = metrics_nms if metrics_nms is not None else metrics_no_nms
                    metrics = metrics_no_nms
                    # early stop/ log / save model
                    metrics_names = ["0.5-r1", "0.5-r10", "0.7-r1", "0.7-r10", "r1", "r10"]
                    for task_type in ["SVMR", "VCMR"]:
                        if task_type in metrics:
                            task_metrics = metrics[task_type]
                            for iou_thd in [0.5, 0.7]:
                                opt.writer.add_scalars("Eval/{}-{}-{}".format(lang, task_type, iou_thd),
                                                       {k: v for k, v in task_metrics.items()
                                                        if str(iou_thd) in k and k in metrics_names},
                                                       global_step)

                    task_type = "VR"
                    if task_type in metrics:
                        task_metrics = metrics[task_type]
                        opt.writer.add_scalars("Eval/{}-{}".format(lang, task_type),
                                               {k: v for k, v in task_metrics.items() if k in metrics_names},
                                               global_step)

            # use the most strict metric available
            stop_metric_names = ["r1"] if opt.stop_task == "VR" else ["0.5-r1", "0.7-r1"]
            stop_score = sum([results[lang]["metrics_no_nms"][opt.stop_task][e]
                              for e in stop_metric_names for lang in languages])

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "model_cfg": model.config,
                    "epoch": epoch_i}
                torch.save(checkpoint, opt.ckpt_filepath)

                latest_file_paths = results["en"]["latest_file_paths"] + results["zh"]["latest_file_paths"]
                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write("Early Stop at epoch {}".format(epoch_i))
                    logger.info("Early stop at {} with {} {}"
                                .format(epoch_i, " ".join([opt.stop_task] + stop_metric_names), prev_best_score))
                    break
        else:
            checkpoint = {
                "model": model.state_dict(),
                "model_cfg": model.config,
                "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

        if opt.debug:
            break

    opt.writer.close()


def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    train_dataset_en, train_eval_dataset_en, eval_dataset_en = \
        get_datasets(opt, lang="en",
                     desc_bert_path_or_handler=opt.desc_bert_path_en,
                     sub_bert_path_or_handler=opt.sub_bert_path_en,
                     vid_feat_path_or_handler=opt.vid_feat_path
                     )

    train_dataset_zh, train_eval_dataset_zh, eval_dataset_zh = \
        get_datasets(opt, lang="zh",
                     desc_bert_path_or_handler=opt.desc_bert_path_zh,
                     sub_bert_path_or_handler=opt.sub_bert_path_zh,
                     vid_feat_path_or_handler=train_dataset_en.vid_feat_h5 if "vid" in opt.ctx_mode else None,
                     )
    datasets = dict(
        train=dict(en=train_dataset_en, zh=train_dataset_zh),
        train_eval=dict(en=train_eval_dataset_en, zh=train_dataset_zh),
        eval=dict(en=eval_dataset_en, zh=eval_dataset_zh)
    )

    # datasets = dict(
    #     train=DatasetShared(en_dataset=train_dataset_en, zh_dataset=train_dataset_zh),
    #     train_eval=DatasetShared(en_dataset=train_eval_dataset_en, zh_dataset=train_eval_dataset_zh),
    #     eval=dict(en=eval_dataset_en, zh=eval_dataset_zh)
    # )

    model_config = EDict(
        # XMLShared config
        share_vid_enc=opt.share_vid_enc,
        share_sub_enc=opt.share_sub_enc,
        share_q_enc=opt.share_q_enc,
        share_pred=opt.share_pred,
        share_q_mapping=opt.share_q_mapping,
        share_input_proj_in_shared_enc=opt.share_input_proj_in_shared_enc,
        # XML config
        merge_two_stream=not opt.no_merge_two_stream,  # merge video and subtitles
        cross_att=not opt.no_cross_att,  # use cross-attention when encoding video and subtitles
        span_predictor_type=opt.span_predictor_type,  # span_predictor_type
        encoder_type=opt.encoder_type,  # gru, lstm, transformer
        add_pe_rnn=opt.add_pe_rnn,  # add pe for RNNs
        pe_type=opt.pe_type,  #
        visual_input_size=opt.vid_feat_size,
        sub_input_size=opt.sub_feat_size,  # for both desc and subtitles
        query_input_size=opt.q_feat_size,  # for both desc and subtitles
        hidden_size=opt.hidden_size,  #
        stack_conv_predictor_conv_kernel_sizes=opt.stack_conv_predictor_conv_kernel_sizes,  #
        conv_kernel_size=opt.conv_kernel_size,
        conv_stride=opt.conv_stride,
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        input_drop=opt.input_drop,
        cross_att_drop=opt.cross_att_drop,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        ctx_mode=opt.ctx_mode,  # video, sub or video_sub
        margin=opt.margin,  # margin for ranking loss
        ranking_loss_type=opt.ranking_loss_type,  # loss type, 'hinge' or 'lse'
        lw_neg_q=opt.lw_neg_q,  # loss weight for neg. query and pos. context
        lw_neg_ctx=opt.lw_neg_ctx,  # loss weight for pos. query and neg. context
        lw_st_ed=0,  # will be assigned dynamically at training time
        lw_neighbors=opt.lw_neighbors,  # loss weight for neighbor constraints
        neighbors_names=opt.neighbors_names,  # which embeddings to apply neighbor constraints
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size,
        use_self_attention=not opt.no_self_att,  # whether to use self attention
        no_modular=opt.no_modular
    )
    logger.info("model_config {}".format(model_config))
    model = XMLShared(model_config)
    count_parameters(model)
    logger.info("Start Training...")
    train(model, opt, datasets)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug


if __name__ == '__main__':
    model_dir, eval_split_name, eval_path, debug = start_training()
    if not debug:
        model_dir = model_dir.split(os.sep)[-1]
        tasks = ["SVMR", "VCMR", "VR"]
        input_args = ["--model_dir", model_dir,
                      "--nms_thd", "0.5",
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path,
                      "--tasks"] + tasks

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model in {}".format(model_dir))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference()
