''' Define the Layers '''
import torch.nn as nn
import torch
import numpy as np
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward,SEQMultiHeadAttention,SEQ2MultiHeadAttention
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
cuda=torch.cuda.is_available()
__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model_i, d_inner, n_head, d_k, d_v, n_position,BATCH_SIZE,dropout=0.1):
        super(EncoderLayer, self).__init__()
        #self.slf_attn = MultiHeadAttention(n_head, d_model_i+d_model_i,d_model_i+d_model_i, d_k, d_v, dropout=dropout)
        #self.lstm = nn.LSTM(d_model_i+d_model_i, d_model_i,batch_first=True,bidirectional=True)
        self.pos_ffn = PositionwiseFeedForward(d_model_i+d_model_i, d_inner, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model_i+d_model_i, d_inner, dropout=dropout)
        #self.position_enc = PositionalEncoding(d_model_i, n_position)
        self.d_model_i=d_model_i
        self.BATCH_SIZE=BATCH_SIZE
        self.sig=nn.Sigmoid()
    def init_hidden(self,device,batch_size):
        # 何かをなす前には、どのような隠れ状態も持ちません。
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        return (torch.zeros(2, batch_size, self.d_model_i).to(device),torch.zeros(2, batch_size, self.d_model_i).to(device))

    def forward(self, enc_input, slf_attn_mask=None):
        #if slf_attn_mask!=None:
            #print("LAY_DEC_MASK",slf_attn_mask.size())
        #enc_input=self.position_enc(enc_input)
        #residual=enc_input
        textlen=enc_input.size(1)

        sentence=pack_padded_sequence(enc_input, slf_attn_mask.cpu(), enforce_sorted=False,batch_first=True)
        #print("MODEL_SENTENCE_PACK",sentence.data.size(),labellen)

        '''
        lstm_out, self.hidden = self.lstm(
            sentence.view(len(sentence), 1, -1), self.hidden)
        '''
        enc_output = self.sig(self.pos_ffn(enc_input))

        enc_output = self.pos_ffn2(enc_output)
        #enc_output=enc_output+residual
        return enc_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model_i,d_model_o, d_inner, n_head, d_k, d_v,videomaxlen,BATCH_SIZE,n_position,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model_o+d_model_o, d_model_o+d_model_o,batch_first=True)
        #self.slf_attn =SEQMultiHeadAttention(n_head, d_model_o+d_model_o,d_model_o+d_model_o, d_k, d_v,videomaxlen,BATCH_SIZE, dropout=dropout)
        self.enc_attn =SEQ2MultiHeadAttention(n_head, d_model_o+d_model_o,d_model_i+d_model_i, d_k, d_v,videomaxlen,BATCH_SIZE, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model_o+d_model_o, d_inner, dropout=dropout)
        #self.position_enc = PositionalEncoding(d_model_i, n_position)
        #self.position_enc2 = IndexPositionalEncoding(d_model_o, n_position)
        self.d_model_o=d_model_o
        self.BATCH_SIZE=BATCH_SIZE
    def init_hidden(self,device,batch_size):
        # 何かをなす前には、どのような隠れ状態も持ちません。
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        return (torch.zeros(1, batch_size, self.d_model_o+self.d_model_o).to(device),torch.zeros(1, batch_size, self.d_model_o+self.d_model_o).to(device))

    def forward(
            self, dec_input, enc_output,index,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        '''
        if dec_enc_attn_mask!=None:
            print("LAY_DEC_MASK",dec_enc_attn_mask.size())
        '''
        #print(index,dec_input.size(),self.position_enc2.pos_table.size(),self.position_enc.pos_table.size())
        #dec_input=self.position_enc2(dec_input,index)
        #residual=dec_input
        if index==0:
            self.hidden=self.init_hidden(dec_input.device,dec_input.size(0))



        lstm_out, self.hidden = self.lstm(dec_input, self.hidden)
        #dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input,index, mask=slf_attn_mask)

        #print("line 93",lstm_out.size(), enc_output.size(), dec_enc_attn_mask.size(),index)
        #dec_output=dec_output+residual
        #enc_output=self.position_enc(enc_output)
        #residual2=enc_output

        dec_output, dec_enc_attn = self.enc_attn(lstm_out, enc_output, enc_output,index, mask=dec_enc_attn_mask)
        #dec_output=dec_output+residual
        dec_output = self.pos_ffn(dec_output)
        #return dec_output[:,:,0:self.d_model_o], dec_slf_attn, dec_enc_attn
        return dec_output, dec_enc_attn
