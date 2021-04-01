from torch import nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange



def multi_head_attention(Q, K, V, mask):
    """Multi-head attention on a batch of Q, K, V
    Arguments:
        Q: torch.Tensor shape (B, N, n_head, d_k)
        K: torch.Tensor shape (B, M, n_head, d_k)
        V: torch.Tensor shape (B, M, n_head, d_v)
        mask: torch.BoolTensor shape (B, N, M)
        where mask[i] is `mask` for attention of record i: (Q[i], K[i], V[i])

    Return:
        scaled-dot attention: torch.Tensor shape (B, N, n_head d_v)
        similar_score: torch.FloatTensor shape (B, n_head, N, M)
    """
    B, N, n_head, d_k = Q.shape
    similar_score = torch.einsum('bnhd,bmhd->bhnm', Q, K) / (d_k ** 0.5)
    similar_score = similar_score.masked_fill(mask.unsqueeze(1), value=float("-inf"))
    attention_weight = F.softmax(similar_score, dim=-1)
    attentions = torch.einsum('bhnm,bmhd->bnhd', attention_weight, V)
    return attentions, similar_score


class Transfomer(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.src_pad_id = hparams['src_pad_id']
        self.tgt_pad_id = hparams['tgt_pad_id']
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.register_buffer("casual_mask", generate_casual_mask(hparams['seq_len']))
        self.apply(self._init_weights)

    def forward(self, src_ids, tgt_ids):
        src_mask = self.make_src_mask(src_ids)
        tgt_mask = self.make_tgt_mask(tgt_ids)
        encoder_output = self.encode(src_ids, src_mask)
        tgt_logits, encoder_attentions = self.decode(encoder_output, src_mask, tgt_ids, tgt_mask)

        return tgt_logits, encoder_attentions

    def make_src_mask(self, src_ids):
        pad_mask = src_ids.eq(self.src_pad_id)
        src_mask = pad_mask.unsqueeze(1).repeat((1, self.hparams['seq_len'], 1))
        return src_mask

    def make_tgt_mask(self, tgt_ids):
        pad_mask = tgt_ids.eq(self.tgt_pad_id)
        tgt_mask = pad_mask.unsqueeze(1).repeat((1, self.hparams['seq_len'], 1))
        return tgt_mask | self.casual_mask

    def encode(self, src_ids, src_mask):
        return self.encoder(src_ids, src_mask)

    def decode(self, encoder_output, src_mask, tgt_ids, tgt_mask):
        tgt_logits, encoder_attentions = self.decoder(encoder_output, src_mask, tgt_ids, tgt_mask)
        return tgt_logits, encoder_attentions

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class Embedding(nn.Module):
    def __init__(self, hparams: dict):
        super(Embedding, self).__init__()
        self.word_embedding = nn.Embedding(hparams['vocab_size'], hparams['d_model'])
        positional_encoding = (
            torch.tensor(
                generate_positional_encoding(hparams['seq_len'], hparams['d_model']), requires_grad=False
            )
                .float()
                .unsqueeze(0)
        )
        self.register_buffer('positional_encoding', positional_encoding)
        self.dropout = nn.Dropout(hparams['embed_dropout'])
        self.scale = hparams['d_model']**0.5

    def forward(self, src_ids: torch.LongTensor):
        """

        """
        x = self.word_embedding(src_ids) * self.scale
        x = x + self.positional_encoding
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.self_attention = AttentionAddNorm(hparams)
        self.feed_forward = FeedForwardAddNorm(hparams)

    def forward(self, src_embeddings, src_masks):
        attention_add_norm, _ = self.self_attention(src_embeddings, src_embeddings, src_embeddings, src_masks)
        ff_add_norm = self.feed_forward(attention_add_norm)

        return ff_add_norm

class Encoder(nn.Module):
    def __init__(self, hparams: dict):

        super(Encoder, self).__init__()
        self.embedding = Embedding(hparams)
        self.blocks = nn.ModuleList([EncoderBlock(hparams) for _ in range(hparams['num_encoder_layers'])])

    def forward(self, src_ids, src_masks):
        src_embed = self.embedding(src_ids)
        for encoder_block in self.blocks:
            src_embed = encoder_block(src_embed, src_masks)
        return src_embed

class DecoderBlock(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.self_attention = AttentionAddNorm(hparams)
        self.cross_attention = AttentionAddNorm(hparams)
        self.feed_forward = FeedForwardAddNorm(hparams)

    def forward(self, encoder_output, src_mask, tgt_embed, tgt_mask):
        tgt_query, tgt_similar_score = self.self_attention(tgt_embed, tgt_embed, tgt_embed, tgt_mask)
        cross_query, cross_similar_score = self.cross_attention(tgt_query, encoder_output, encoder_output, src_mask)
        embedding = self.feed_forward(cross_query)
        encoder_attention = cross_similar_score

        return embedding, encoder_attention

class Decoder(nn.Module):
    def __init__(self, hparams: dict):
        super(Decoder, self).__init__()
        self.embedding = Embedding(hparams)
        self.blocks = nn.ModuleList([DecoderBlock(hparams) for _ in range(hparams['num_decoder_layers'])])
        self.output_layer = nn.Linear(hparams['d_model'], hparams['tgt_vocab_size'])

    def forward(self, encoder_output, src_mask, tgt_ids, tgt_casual_mask):
        encoder_attentions = []
        tgt_embed = self.embedding(tgt_ids)
        for decoder_block in self.blocks:
            tgt_embed, encoder_attention = decoder_block(encoder_output, src_mask, tgt_embed, tgt_casual_mask)
            encoder_attentions.append(encoder_attention)
        logits = self.output_layer(tgt_embed)

        return logits, encoder_attentions

class FeedForwardAddNorm(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.linear1 = nn.Linear(hparams['d_model'], hparams['d_ff'])
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hparams['d_ff'], hparams['d_model'])
        self.dropout = nn.Dropout(hparams['ff_dropout'])
        self.layer_norm = nn.LayerNorm(hparams['d_model'])

    def forward(self, input):
        x = self.linear1(input)
        x = self.activation(x)
        x = self.linear2(x)
        x = input + self.dropout(x)
        x = self.layer_norm(x)
        return x




class AttentionAddNorm(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams=hparams
        self.Q_proj = nn.Linear(self.hparams['d_model'], self.hparams['d_model'])
        self.K_proj = nn.Linear(self.hparams['d_model'], self.hparams['d_model'])
        self.V_proj = nn.Linear(self.hparams['d_model'], self.hparams['d_model'])
        self.dropout = nn.Dropout(self.hparams['attn_dropout'])
        self.layer_norm = nn.LayerNorm(self.hparams['d_model'])

    def forward(self, query, key, value, mask):
        Q, K, V = self.Q_proj(query), self.K_proj(key), self.V_proj(value)
        Q = rearrange(Q, "B seq_len (n_head d_k) -> B seq_len n_head d_k", n_head=self.hparams['n_head'])
        K = rearrange(K, "B seq_len (n_head d_k) -> B seq_len n_head d_k", n_head=self.hparams['n_head'])
        V = rearrange(V, "B seq_len (n_head d_v) -> B seq_len n_head d_v", n_head=self.hparams['n_head'])
        attention, similar_score = multi_head_attention(Q, K, V, mask)
        attention = self.dropout(attention)
        attention = rearrange(attention, "B seq_len n_head d_v -> B seq_len (n_head d_v)")
        return self.layer_norm(query + attention), similar_score




def generate_positional_encoding(seq_len, d_model):
    """
    """
    pe = torch.zeros((seq_len, d_model))

    position = torch.arange(0, seq_len).unsqueeze(1)
    denominator = torch.exp(math.log(10000.0)*(-torch.arange(0, d_model, 2))/d_model)
    pe[:, 0::2] = torch.sin(position*denominator)
    pe[:, 1::2] = torch.cos(position*denominator)

    return pe

def generate_casual_mask(size):
    """

    """
    mask = torch.ones((size, size))
    return torch.triu(mask, diagonal=1).bool()



