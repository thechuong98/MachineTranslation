import pytorch_lightning as pl
import torch.nn.functional as F
from datasets import load_metric
import torch
import hydra
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = load_metric('sacrebleu')


    def forward(self, src_ids, tgt_ids):
        tgt_inp_ids = tgt_ids[:, :-1]
        tgt_out_ids = tgt_ids[:, 1:]
        logits, encoder_attentions = self.model(src_ids, tgt_inp_ids)
        return logits

    def step(self, batch):
        src_ids, tgt_ids = batch
        logits = self.forward(src_ids, tgt_ids)
        loss = self.criterion(logits.view(self.hparams['batch_size'], self.hparams['tgt_vocab_size'], -1), tgt_ids[:, 1:])
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss.detach().cpu().item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss.detach().cpu().item())
        return loss

    def test_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        logits = self.model(src_ids, tgt_ids)

    def configure_optimizers(self):
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        return optim
