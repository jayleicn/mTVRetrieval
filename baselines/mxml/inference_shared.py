import pprint
import torch
import torch.backends.cudnn as cudnn
from baselines.mxml.config_shared import TestOptions
from baselines.mxml.model_xml_shared import XMLShared
from baselines.xml.inference import eval_epoch, setup_model, get_eval_dataset


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    assert opt.eval_path is not None
    eval_dataset_en = get_eval_dataset(
        opt, lang="en",
        desc_bert_path_or_handler=opt.desc_bert_path_en,
        sub_bert_path_or_handler=opt.sub_bert_path_en,
        vid_feat_path_or_handler=opt.vid_feat_path
    )
    eval_dataset_zh = get_eval_dataset(
        opt, lang="zh",
        desc_bert_path_or_handler=opt.desc_bert_path_zh,
        sub_bert_path_or_handler=opt.sub_bert_path_zh,
        vid_feat_path_or_handler=eval_dataset_en.vid_feat_h5 if "vid" in opt.ctx_mode else None,  # share video
    )
    eval_datasets = dict(en=eval_dataset_en, zh=eval_dataset_zh)

    model = setup_model(opt, model_class=XMLShared)
    logger.info("Starting inference...")
    for lang, eval_dataset in eval_datasets.items():
        save_submission_filename = "{}_inference_{}_{}_{}_predictions_{}.json".format(
            lang, opt.dset_name, opt.eval_split_name, opt.eval_id, "_".join(opt.tasks))
        with torch.no_grad():
            _model = getattr(model, f"xml_{lang}")
            metrics_no_nms, metrics_nms, latest_file_paths = \
                eval_epoch(_model, eval_dataset, opt, save_submission_filename,
                           tasks=opt.tasks, max_after_nms=100)
        logger.info("{} metrics_no_nms \n{}".format(lang, pprint.pformat(metrics_no_nms, indent=4)))
        logger.info("{} metrics_nms \n{}".format(lang, pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
