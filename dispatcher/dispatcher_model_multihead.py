import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones


class DispatcherLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, bptt, dropout=0.):
        super(DispatcherLayer, self).__init__()

        self._levels = int(math.log(bptt, 2))
        self._shifts = [pow(2, i) for i in range(self._levels)]

        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.linear_in = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.internal_attention = nn.Linear(embed_dim, self._levels * num_heads, bias=False)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)


    def forward(self, value, attn_mask):
        inp = value.transpose(1, 0)
        batch_length = inp.shape[0]
        length = inp.shape[1]

        V = self.linear_in(inp)

        attention = self.internal_attention(inp)
        attention = attention.reshape(batch_length * self.num_heads, length, self._levels)
        attention *= attn_mask.detach()
        attentions = torch.chunk(attention, chunks=self._levels, dim=2)

        V = V.reshape(batch_length * self.num_heads, length, self.head_dim)

        for a, shift in zip(attentions, self._shifts):
            if shift > length:
                break
            if shift > 1 and self.training and random.uniform(0, 1) < self.dropout:
                continue
            V += a * torch.roll(V, shifts=shift, dims=1)

        V = V.reshape(batch_length, length, self.embed_dim)
        out = self.linear_out(V)
        return out.transpose(1, 0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, bptt, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = DispatcherLayer(d_model, nhead, bptt, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DispatcherModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, bptt, dropout=0.5):
        super(DispatcherModel, self).__init__()
        self._levels = int(math.log(bptt, 2))
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, bptt, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, length):
        mask = torch.zeros([length, self._levels], requires_grad=False)

        for i in range(self._levels):
            shift = pow(2, i)
            mask[shift:, i] = torch.ones_like(mask[shift:, i], requires_grad=False)

        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
