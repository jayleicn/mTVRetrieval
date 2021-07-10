"""
Dataset for clip model
"""
import logging
from torch.utils.data import Dataset
from baselines.xml.start_end_dataset import start_end_collate


logger = logging.getLogger(__name__)


class DatasetShared(Dataset):
    """https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
    """
    def __init__(self, en_dataset, zh_dataset):
        # the two datasets are the same, except for the language.
        self.en_dataset = en_dataset
        self.zh_dataset = zh_dataset

    def __getitem__(self, index):
        return dict(
            en=self.en_dataset[index],
            zh=self.zh_dataset[index]
        )

    def __len__(self):
        return len(self.en_dataset)


def start_end_collate_shared(batch):
    return start_end_collate([b["en"] for b in batch]), \
           start_end_collate([b["zh"] for b in batch])

