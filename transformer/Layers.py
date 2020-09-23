''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
# import pdb

__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)


    def forward(self, enc_input, slf_attn_mask=None):
        enc_input2 = self.layer_norm1(enc_input)
        # pdb.set_trace()
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input2, enc_input2, enc_input2, mask=slf_attn_mask)
        enc_output = enc_input + enc_output
        enc_output2 = self.layer_norm2(enc_output)
        enc_output2 = self.pos_ffn(enc_output2)
        enc_output = enc_output + enc_output2
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_input2 = self.layer_norm1(dec_input)
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input2, dec_input2, dec_input2, mask=slf_attn_mask)
        dec_output = dec_input + dec_output

        dec_output2 = self.layer_norm2(dec_output)
        dec_output2, dec_enc_attn = self.enc_attn(
            dec_output2, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = dec_output + dec_output2
        dec_output2 = self.layer_norm3(dec_output)
        dec_output2 = self.pos_ffn(dec_output2)
        dec_output = dec_output + dec_output2

        return dec_output, dec_slf_attn, dec_enc_attn
