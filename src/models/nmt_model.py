import pytorch_lightning as pl
import torch.nn.functional as F
from datasets import load_metric
import torch

from src.architectures.transformer import Transfomer
from pytorch_lightning.metrics.classification import Accuracy, F1


class NMT_LitModel(pl.LightningModule):
    def __init__(self, hparams, src_tokenizer, tgt_tokenizer):
        super(NMT_LitModel, self).__init__()
        self.hparams = hparams
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.hparams['src_pad_id'] = src_tokenizer.get_vocab()['[PAD]']
        self.hparams['tgt_pad_id'] = tgt_tokenizer.get_vocab()['[PAD]']
        self.model = Transfomer(hparams)

    def forward(self, src_ids, tgt_ids):
        tgt_inp_ids = tgt_ids[:, :-1]
        tgt_out_ids = tgt_ids[:, 1:]
        logits, encoder_attentions = self.model(src_ids, tgt_inp_ids)
        loss = F.cross_entropy(logits, tgt_out_ids)

        return loss

    def training_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        loss = self.forward(src_ids, tgt_ids)
        self.log('train_loss', loss.detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        loss = self.forward(src_ids, tgt_ids)
        self.log('val_loss', loss.detach().cpu().item())
        return loss



    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
        return optimizer
