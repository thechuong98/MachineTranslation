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
from einops import rearrange


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
        # self.criterion = torch.nn.functional.cross_entropy()
        self.metric = load_metric('sacrebleu')
        self.accuracy = Accuracy()
        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }


    # def forward(self, src_ids, tgt_ids):
    #     tgt_inp_ids = tgt_ids[:, :-1]
    #     tgt_out_ids = tgt_ids[:, 1:]
    #     logits, encoder_attentions = self.model(src_ids, tgt_inp_ids)
    #     return logits, tgt_out_ids, tgt_inp_ids

    def forward(self, src_ids, tgt_ids):
        # src_ids: B x seq_len
        # tgt_ids: B x (seq_len + 1) for teacher-forcing
        ########### YOUR CODE HERE #################
        tgt_inp_ids = tgt_ids[:, :-1]
        tgt_out_tokens = tgt_ids[:, 1:]
        logits, encoder_attentions = self.model(src_ids, tgt_inp_ids)
        loss = F.cross_entropy(rearrange(logits, "B seq_len vocab_size -> B vocab_size seq_len"), tgt_out_tokens)
        ############################################
        return loss

    # def step(self, batch):
    #     src_ids, tgt_ids = batch
    #     logits, tgt_out_ids, tgt_inp_ids = self.forward(src_ids, tgt_ids)
    #     loss = F.cross_entropy(logits.view(self.hparams['batch_size'], self.hparams['tgt_vocab_size'], -1), tgt_ids[:, 1:])
    #     return loss, tgt_out_ids, tgt_inp_ids


    def training_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        loss = self.forward(src_ids, tgt_ids)
        self.log('train/loss', loss)
        self.log('train/acc', torch.Tensor([0]))
        return loss

    def validation_step(self, batch, batch_idx):
        src_ids, tgt_ids = batch
        loss = self.forward(src_ids, tgt_ids)
        self.log('val/loss', loss.detach().cpu().item())
        self.log('val/acc', torch.Tensor([0]))
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
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.model.parameters()
        )
        return optim
