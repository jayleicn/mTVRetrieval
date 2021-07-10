import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from easydict import EasyDict as edict
from baselines.xml.model_xml import XML


xml_shared_config = edict(
    # the configs that already exist in xml_base_config is ignored here.
    share_vid_enc=False,  # share video encoder
    share_sub_enc=False,  # share sub encoder
    share_q_enc=False,  # share query encoder
    share_pred=False,  # share the final st_ed predictors
    share_q_mapping=False,  # share modular query and ...
    share_input_proj_in_shared_enc=False,  # share *_input_proj layers in the encoders that are shared
    lw_neighbors=1,  # loss weight for neighbor constraints
    neighbors_names=["query", "sub1", "sub2"]  # which embeddings to apply neighbor constraints
)


class XMLShared(nn.Module):
    """Create two XML models that share part of the parameters."""
    xml_vid_enc_module_names = [
        "ctx_pos_embed", "video_input_proj",
        "video_encoder1", "video_encoder2", "video_encoder3",
    ]

    xml_sub_enc_module_names = [
        "ctx_pos_embed", "sub_input_proj",
        "sub_encoder1", "sub_encoder2", "sub_encoder3",
    ]

    xml_q_enc_module_names = [
        "query_pos_embed", "query_input_proj", "query_encoder",
    ]

    xml_q_modality_mapping_module_names = [
        "modular_vector_mapping", "video_query_linear", "sub_query_linear"
    ]

    xml_st_ed_predictor_module_names = [
        "merged_st_predictor", "merged_ed_predictor"
    ]

    def __init__(self, config):
        super(XMLShared, self).__init__()
        self.config = config
        self.languages = ["en", "zh"]
        self.xml_en = XML(config)
        self.xml_zh = XML(config)

        # sharing weight
        self.shared_module_names = self.get_shared_module_names()
        strict = False
        self.tie_weights_or_check_same_by_top_module_names(
            self.shared_module_names, strict=strict, mode="tie")
        # check shared
        self.tie_weights_or_check_same_by_top_module_names(
            self.shared_module_names, strict=strict, mode="check")

    def forward(self, en_zh_inputs):
        """
        Args:
            en_zh_inputs: a dict containing dict(en=en_data_dict, zh=zh_data_dict)
        """
        loss = 0
        loss_dicts = {}
        neighbor_embeddings = {}
        for lang in self.languages:
            _model = getattr(self, f"xml_{lang}")
            en_zh_inputs[lang]["return_neighbor_embeddings"] = True
            _loss, _loss_dict, _neighbor_embeddings = _model(**en_zh_inputs[lang])
            loss += _loss
            loss_dicts[lang] = _loss_dict
            neighbor_embeddings[lang] = _neighbor_embeddings

        # cross language alignment loss
        if self.config.lw_neighbors > 0:
            _loss_lang_alignment = self.compute_neighbor_constraint_loss(neighbor_embeddings)
            loss += self.config.lw_neighbors * _loss_lang_alignment
            # add in only en dict for simplicity, this loss computes across languages
            loss_dicts["en"]["lang_align"] = float(_loss_lang_alignment)

        return loss, loss_dicts

    def compute_neighbor_constraint_loss(self, neighbor_embeddings):
        assert self.config.lw_neighbors > 0
        neighbors_names = self.config.neighbors_names
        neighbor_loss = 0
        if "query" in neighbors_names:
            for query_name in ["video_query", "sub_query"]:
                zh_matrix = neighbor_embeddings["zh"][query_name]  # (N, D)
                en_matrix = neighbor_embeddings["en"][query_name]  # (N, D)
                if zh_matrix is not None and en_matrix is not None:
                    neighbor_loss += self.get_neighbor_loss_for_pair(zh_matrix, en_matrix)
        for sub_name in ["sub1", "sub2"]:
            if sub_name in neighbors_names:
                zh_matrix = neighbor_embeddings["zh"][sub_name]
                en_matrix = neighbor_embeddings["en"][sub_name]
                if zh_matrix is not None and en_matrix is not None:
                    neighbor_loss += self.get_neighbor_loss_for_pair(zh_matrix, en_matrix)
        return neighbor_loss

    def get_neighbor_loss_for_pair(self, embedding1, embedding2):
        """
        Args:
            embedding1: (N, D)
            embedding2: (N, D)
        """
        embedding1 = F.normalize(embedding1, dim=-1)
        embedding2 = F.normalize(embedding2, dim=-1)
        cross_lang_scores = torch.einsum(
            "md,nd->mn", embedding1, embedding2)  # (N1, N2)
        # note the self.xml_en.get_video_level_loss is the same as self.xml_zh.get_video_level_loss
        loss_neg2, loss_neg1 = self.xml_en.get_video_level_loss(cross_lang_scores)
        return loss_neg2 + loss_neg1

    def forward_en(self, **kwargs):
        self.xml_en.forward(**kwargs)

    def forward_zh(self, **kwargs):
        self.xml_zh.forward(**kwargs)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        self.xml_en.set_hard_negative(use_hard_negative, hard_pool_size)
        self.xml_zh.set_hard_negative(use_hard_negative, hard_pool_size)

    def set_train_st_ed(self, lw_st_ed):
        self.xml_en.set_train_st_ed(lw_st_ed)
        self.xml_zh.set_train_st_ed(lw_st_ed)

    def get_shared_module_names(self):
        """
    share_vid_enc=False,  # share video encoder
    share_sub_enc=False,  # share sub encoder
    share_q_enc=False,  # share query encoder
    share_pred=False, # share the final st_ed predictors
    share_q_mapping=False, #
    share_input_proj_in_shared_enc=False,  # share *_input_proj layers in the encoders that are shared
        Returns:

        """
        shared_module_names = []
        # adding module names
        if self.config.share_vid_enc:
            shared_module_names += self.xml_vid_enc_module_names
        if self.config.share_sub_enc:
            shared_module_names += self.xml_sub_enc_module_names
        if self.config.share_q_enc:
            shared_module_names += self.xml_q_enc_module_names
        if self.config.share_pred:
            shared_module_names += self.xml_st_ed_predictor_module_names
        if self.config.share_q_mapping:
            shared_module_names += self.xml_q_modality_mapping_module_names

        if not self.config.share_input_proj_in_shared_enc:
            shared_module_names = [
                name for name in shared_module_names if "input_proj" not in name]
        return shared_module_names

    # def tie_weights_or_check_same_video_encoder(self, mode: str = "tie"):
    #     _module_names = self.xml_video_encoding_related_module_names
    #     if not self.config.share_input_proj:
    #         do_not_tie = ["video_input_proj"]
    #         module_names = [name for name in _module_names if name not in do_not_tie]
    #     else:
    #         module_names = _module_names
    #     self.tie_weights_or_check_same_by_top_module_names(module_names, strict=True, mode=mode)
    #
    # def tie_weights_or_check_same_all_encoder(self, mode: str = "tie"):
    #     _module_names = self.xml_video_encoding_related_module_names + \
    #         self.xml_query_encoding_module_names + \
    #         self.xml_sub_encoding_related_module_names + \
    #         self.xml_non_query_shared_module_names
    #     if not self.config.share_input_proj:
    #         do_not_tie = ["video_input_proj", "query_input_proj", "sub_input_proj"]
    #         module_names = [name for name in _module_names if name not in do_not_tie]
    #     else:
    #         module_names = _module_names
    #     self.tie_weights_or_check_same_by_top_module_names(module_names, strict=True, mode=mode)

    def tie_weights_or_check_same_by_top_module_names(
            self, module_names: List[str], strict: bool = False, mode: str = "tie"):
        """ Tie the weight between self.xml_en and self.xml_zh
        Args:
            module_names: list(str), each str represents a module name defined at one of the XML models
            strict: bool, if True, all the modules_names must exist in the XML models
            mode: str, tie or check
        """
        module1 = self.xml_zh
        module2 = self.xml_en
        non_exist_module_names = []
        for name in module_names:
            if hasattr(module1, name):
                assert hasattr(module2, name)
                self._tie_weights_or_check_same_recursively(
                    getattr(module1, name), getattr(module2, name), mode=mode)
            else:
                non_exist_module_names.append(name)

        if strict:
            assert len(non_exist_module_names) == 0, \
                f"Strict mode, all module names have to exist in the modules, " \
                f"but got nonexist {non_exist_module_names}"
        else:
            if len(non_exist_module_names) != 0:
                print(f"Non strict mode, there are {len(non_exist_module_names)} "
                      f"names not found: {non_exist_module_names}")

    def _tie_weights_or_check_same_recursively(self, module1: nn.Module, module2: nn.Module, mode: str = "tie"):
        """Tie weights/bias or check whether weights/bias are the same between module1 and module2, recursively.
        Order does not matter. We assume module1 and module2 are initialized from the exact same class.
        mode: str,
            tie: Tie weights/bias
            check: check whether the weights/bias are the same using torch.allclose()
        ref: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/modeling_utils.py
        """
        assert module1.__class__ == module2.__class__, \
            f"{module1.__class__} and {module2.__class__} does not equal. " \
            f"We assume module1 and module2 are initialized from the exact same class."
        assert isinstance(module1, nn.Module) and isinstance(module2, nn.Module), \
            f"{module1} and {module2} have to be of type torch.nn.Module"
        assert mode in ["tie", "check"]

        if hasattr(module1, "weight") and getattr(module1, "weight") is not None:  # has attr and is not None
            assert hasattr(module2, "weight")
            if mode == "tie":
                module1.weight = module2.weight
            elif mode == "check":
                assert torch.allclose(module1.weight, module2.weight), f"{module1} weights != {module2} weights."
            if hasattr(module1, "bias") and getattr(module1, "bias") is not None:
                assert hasattr(module2, "bias")
                if mode == "tie":
                    module1.bias = module2.bias
                elif mode == "check":
                    assert torch.allclose(module1.bias, module2.bias), f"{module1} bias != {module2} bias."
            return

        module1_modules = module1._modules  # OrderedDict
        module2_modules = module2._modules
        if len(module1_modules) > 0:
            assert len(module2_modules) > 0, f"module {module1} and {module2} does not match"

            for name in module1_modules:
                self._tie_weights_or_check_same_recursively(
                    module1_modules[name], module2_modules[name], mode=mode)

