''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_i,d_model_o, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_i, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_o, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_o, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_i, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model_i, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        #if(True):
            #print("sublayer32,",q.size(),k.size(),v.size())


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q.clone()

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q)
        #print("q in line 55",q.size())
        q += residual
        return q, attn
        '''
        q = self.dropout(self.fc(q))
        q =q+ residual

        q = self.layer_norm(q)

        return q, attn
        '''

class SEQMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_i,d_model_o, d_k, d_v,videomaxlen,BATCH_SIZE, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = nn.Linear(d_model_i, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_o, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_o, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_i, bias=False)
        self.BATCH_SIZE=BATCH_SIZE

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model_i, eps=1e-6)
        self.videomaxlen=videomaxlen
        self.register_buffer('zerotable', torch.zeros(BATCH_SIZE,n_head,videomaxlen,d_k))
        self.zerotable=self.zerotable.to(self.w_vs.weight.device)



    def forward(self, q, k, v,index, mask=None):
        #if(True):
            #print("sublayer32,",q.size(),k.size(),v.size())


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q.clone()
        #print("SUB99,q,k,v",q.size(),k.size(),v.size())
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask=mask.to(self.w_vs.weight.device)  # For head axis broadcasting. then, batch 1 seq_len seq_len?
            #print("SUB102",mask.size())

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if index==0:

            #self.q_stock=torch.empty(self.BATCH_SIZE,n_head,0,d_k)
            self.k_stock=torch.empty(sz_b,n_head,0,d_k)
            self.v_stock=torch.empty(sz_b,n_head,0,d_v)

            self.k_stock=self.k_stock.to(self.w_vs.weight.device)
            self.v_stock=self.v_stock.to(self.w_vs.weight.device)

        self.k_stock =torch.cat([self.k_stock ,self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)],dim=2)
        #self.q_stock= torch.cat([self.q_stock ,self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)],dim=2)
        q=self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        self.v_stock= torch.cat([self.v_stock ,self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)],dim=2)

        # Transpose for attention dot product: b x n x lq x dv
        k_use=torch.cat([self.k_stock,self.zerotable[:sz_b,:,index+1:]],dim=2)
        v_use=torch.cat([self.v_stock,self.zerotable[:sz_b,:,index+1:]],dim=2)
        #print("usesize",k_use.size(),v_use.size(),mask[:,:,index:index+1].size())
        #q_elem, attn = self.attention(self.q_stock[:,:,index:index+1],k_use, v_use, mask=mask[:,:,index:index+1])
        q, attn = self.attention(q,k_use, v_use, mask=mask[:,:,index:index+1])
        del k_use
        del v_use

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        q = self.dropout(self.fc(q))
        q = self.layer_norm(q)
        #print("q in line 55",q.size())
        q += residual
        return q, attn


class SEQ2MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_i,d_model_o, d_k, d_v,videomaxlen,BATCH_SIZE, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_i, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_o, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_o, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_i, bias=False)
        self.videomaxlen=videomaxlen



        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.BATCH_SIZE=BATCH_SIZE
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model_i, eps=1e-6)


    def forward(self, q, k, v,index, mask=None):
        #if(True):
            #print("sublayer32,",q.size(),k.size(),v.size())


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), 1, k.size(1), v.size(1)

        residual = q.clone()
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting. then, batch 1 seq_len seq_len?
            mask=mask.to(self.w_vs.weight.device)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if index==0:
            self.k_stock = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
            #self.q_stock=torch.empty(self.BATCH_SIZE,n_head,0,d_k)
            #self.q_stock= torch.cat([self.q_stock ,self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)],dim=2)
            self.v_stock = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)
        #self.q_stock= torch.cat([self.q_stock ,self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)],dim=2)
        q=self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        #print('line196',q.size())
        # Transpose for attention dot product: b x n x lq x dv
        #print("SUB170",self.q_stock[:,:,index:index+1].size(),self.k_stock.size(), self.v_stock.size(), mask.size())
        #q_elem, attn = self.attention(self.q_stock[:,:,index:index+1],self.k_stock, self.v_stock, mask=mask[:,:,index:index+1])
        q, attn = self.attention(q,self.k_stock, self.v_stock, mask=mask)
        #print('line201',q.size(),self.k_stock.size(), self.v_stock.size())

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q= q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #print('line204',q.size(),self.d_v, self.n_head)

        q = self.dropout(self.fc(q))
        q = self.layer_norm(q)
        #print("q in line 55",q.size())
        q += residual
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x.clone()

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x =x+ residual

        x = self.layer_norm(x)

        return x
