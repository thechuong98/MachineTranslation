import pytorch_lightning as pl
import torch.nn.functional as F
from datasets import load_metric
import torch

from src.architectures.transformer import Transfomer
from pytorch_lightning.metrics.classification import Accuracy, F1


class NMTLitModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.src_tokenizer = src_tokenizer
        # self.tgt_tokenizer = tgt_tokenizer
        # self.hparams['src_pad_id'] = src_tokenizer.get_vocab()['[PAD]']
        # self.hparams['tgt_pad_id'] = tgt_tokenizer.get_vocab()['[PAD]']
        self.model = Transfomer(hparams=self.hparams)

    def forward(self, src_ids, tgt_ids):
        tgt_inp_ids = tgt_ids[:, :-1]
        tgt_out_ids = tgt_ids[:, 1:]
        logits, encoder_attentions = self.model(src_ids, tgt_inp_ids)
        return logits

    def training_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        logits = self.forward(src_ids, tgt_ids)
        loss = F.cross_entropy(logits.view(self.hparams['batch_size'], self.hparams['tgt_vocab_size'], -1), tgt_ids[:, 1:])
        self.log('train_loss', loss.detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        logits = self.forward(src_ids, tgt_ids)
        loss = F.cross_entropy(logits.view(self.hparams['batch_size'], self.hparams['tgt_vocab_size'], -1),
                               tgt_ids[:, 1:])
        self.log('val_loss', loss.detach().cpu().item())
        return loss

    def test_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        logits = self.model(src_ids, tgt_ids)

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
