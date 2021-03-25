from torch import nn
import torch
import torch.nn.functional as F
import math

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
        self.positional_encoding = torch.Tensor(
            generate_positional_encoding(hparams['seq_len'], hparams['d_model']),
            requires_grad=False).float().unqueeze(0)
        self.register_buffer('positional_encoding', self.positional_encoding)
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

    def __init__(self, hparams: dict):
        super(FeedForwardAddNorm, self).__init__()
        self.linear1 = nn.Linear(hparams['d_model'], hparams['d_ff'])
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hparams['d_ff'], hparams['d_model'])
        self.dropout = nn.Dropout(hparams['ff_dropout'])
        self.layer_norm = nn.LayerNorm(hparams['d_model'])

    def forward(self, input):
        x_init = input
        x = self.linear1(x_init)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + x_init

        return self.layer_norm(x)


class AttentionAddNorm(nn.Module):
    def __init__(self, hparams: dict):
        self.hparams = hparams
        super(AttentionAddNorm, self).__init__()
        self.Q_proj = nn.Linear(hparams['d_model'], hparams['d_model'])
        self.K_proj = nn.Linear(hparams['d_model'], hparams['d_model'])
        self.V_proj = nn.Linear(hparams['d_model'], hparams['d_model'])
        self.dropout = nn.Dropout(hparams['attn_dropout'])
        self.layer_norm = nn.LayerNorm(hparams['d_model'])

    def forward(self, query, key, value, mask):
        B, seq_len, d_model = query.shape
        init_query = query
        query = self.Q_proj(query).reshape(B, seq_len, self.hparams['d_k'], self.hparams['n_head'])
        key = self.K_proj(key).reshape(B, seq_len, self.hparams['d_k'], self.hparams['n_head'])
        value = self.V_proj(value).reshape(B, seq_len, self.hparams['d_v'], self.hparams['n_head'])
        multihead_attention, similar_score = multihead_attention(query, key, value, mask)
        multihead_attention_dropout = self.dropout(multihead_attention)
        multihead_attention_dropout = multihead_attention_dropout.reshape(B, seq_len, -1)
        add_layer = multihead_attention_dropout + init_query
        norm = self.layer_norm(add_layer)

        return norm

def multihead_attention(Q, K, V, mask):
    """
    Q: (B, N, n_head, d_k)
    K: (B, N, n_head, d_k)
    V: (B, M, n_head, d_v)
    mask (B, N, M)
    """
    B, N, n_head, d_k = Q.shape
    scale = d_k**0.5
    similar_score = torch.einsum('bnhd,bmhd->hbnm', Q, K)*scale
    similar_score = similar_score.masked_fill(mask, value=float('-inf'))
    attention_w = F.softmax(similar_score, dim=-1)
    attention = torch.einsum('hbnm,bmhd->bnhd', attention_w, V)

    return attention, similar_score

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


