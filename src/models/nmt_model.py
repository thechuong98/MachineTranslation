import pytorch_lightning as pl
import torch.nn.functional as F
from datasets import load_metric
import torch
import hydra
from src.architectures.transformer import Transfomer
import os
from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from pytorch_lightning.metrics.classification import Accuracy, F1


class NMTLitModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer_dir = os.path.join(self.hparams['work_dir'], 'script', 'tokenizer')
        self.src_tokenizer = Tokenizer.from_file(os.path.join(self.tokenizer_dir, 'en.json'))
        self.tgt_tokenizer = Tokenizer.from_file(os.path.join(self.tokenizer_dir, 'vi.json'))
        # self.src_pad_id = self.src_tokenizer.get_vocab()['[PAD]']
        # self.tgt_pad_id = self.tgt_tokenizer.get_vocab()['[PAD]']
        self.model = Transfomer(hparams=self.hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = load_metric('sacrebleu')
        self.accuracy = Accuracy()
        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }


    def forward(self, src_ids, tgt_ids):
        tgt_inp_ids = tgt_ids[:, :-1]
        tgt_out_ids = tgt_ids[:, 1:]
        logits, encoder_attentions = self.model(src_ids, tgt_inp_ids)
        return logits, tgt_out_ids, tgt_inp_ids

    def step(self, batch):
        src_ids, tgt_ids = batch
        logits, tgt_out_ids, tgt_inp_ids = self.forward(src_ids, tgt_ids)
        loss = self.criterion(logits.view(self.hparams['batch_size'], self.hparams['tgt_vocab_size'], -1), tgt_ids[:, 1:])
        return loss, tgt_out_ids, tgt_inp_ids


    def training_step(self, batch, batch_idx):
        loss, tgt_out_ids, tgt_inp_ids = self.step(batch)
        acc = self.accuracy(tgt_out_ids, tgt_inp_ids)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, tgt_out_ids, tgt_inp_ids = self.step(batch)
        acc = self.accuracy(tgt_out_ids, tgt_inp_ids)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss


    def configure_optimizers(self):
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        return optim
