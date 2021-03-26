from pytorch_lightning import LightningDataModule
from src.datasets.nmt_dataset import NMTDataset
from torch.utils.data import DataLoader
import os
import sys
sys.path.append('')


ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = 'data/processed_iwslt_data/'

class NMTDataModule(LightningDataModule):
    """

    """

    def __init__(self, *args, **kwargs):
        super(NMTDataModule, self).__init__()
        self.data_dir = os.path.join(kwargs['data_dir'], 'processed_iwslt_data')
        self.batch_size = kwargs['batch_size']
        self.num_worker = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        self.data_train = NMTDataset(self.data_dir, 'train')
        self.data_val = NMTDataset(self.data_dir, 'val')
        self.data_test = NMTDataset(self.data_dir, 'test')


    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          pin_memory=self.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          pin_memory=self.pin_memory,
                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          pin_memory=self.pin_memory,
                          shuffle=True)
