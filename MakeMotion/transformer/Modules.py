import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        #print("before_attn",attn.size(),q.size(),k.size(),v.size())

        if mask is not None:
            #print("MODULEMASK",mask.size())
            attn = attn.masked_fill(mask == 0, -1e9)

        #attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        #print("Module25attn_array,",attn.size())
        output = torch.matmul(attn, v)
        #print("after_attn",attn.size(),mask.size(),output.size())
        '''
        if(True):
            print("Modules.py",output.size(),attn.size())
        '''

        return output, attn
