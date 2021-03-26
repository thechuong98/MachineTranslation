from torch.utils.data import Dataset
from os.path import join as pjoin
import json
import torch
from script.prepare_data import tokenize_data

class NMTDataset(Dataset):
    def __init__(self, root: str, data_type: str):
        super(NMTDataset, self).__init__()
        if not data_type:
            self.src = json.load(open(pjoin(root, 'train_src.json'), encoding='utf-8'))
            self.tgt = json.load(open(pjoin(root, 'train_tgt.json'), encoding='utf-8'))
        self.src = json.load(open(pjoin(root, data_type+'_src.json'), encoding='utf-8'))
        self.tgt = json.load(open(pjoin(root, data_type+'_tgt.json'), encoding='utf-8'))

    def __len__(self):
        return int(len(self.src))

    def __getitem__(self, idx):
            return [torch.LongTensor(self.src[idx]['token_ids']), torch.LongTensor(self.tgt[idx]['token_ids'])]

