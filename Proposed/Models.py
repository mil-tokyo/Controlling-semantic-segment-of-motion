seed=1
D_innerEmbtoData=512
lambda_KL=0.05
import os
os.environ["OMP_NUM_THREADS"] = "1"
D_model_f=32
lambda_r=5
gamma = 0.01
lambda_proc=0
import torch
from DPsoftminLoss import DPsoftminLoss,P_PENAL
import ReconLoss
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
import sys
gpu_ids= [(int)(sys.argv[1])]
BATCH_SIZE=50
bonelist=['mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head','mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm','mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm','mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg']


import torch
import random
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)
import torch.nn as nn
from transformer.Layers import EncoderLayer, DecoderLayer
import dataload
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import detect_anomaly
writer = SummaryWriter()
CLIP_MODEL=100
num_train=8000
lambda_a=1


training_data = dataload.Text2MotionDataset("../../../dataset/HumanML3D/HumanML3D/train.txt")
eval_data = dataload.Text2MotionDataset( "../../../dataset/HumanML3D/HumanML3D/val.txt")
test_data = dataload.Text2MotionDataset("../../../dataset/HumanML3D/HumanML3D/test.txt")
train_loader = torch.utils.data.DataLoader(dataset=training_data,
                         batch_size=BATCH_SIZE,num_workers=2, shuffle=True,worker_init_fn=seed_worker,generator=g,)
eval_loader = torch.utils.data.DataLoader(dataset=eval_data,
                         batch_size=BATCH_SIZE, num_workers=2, shuffle=False,worker_init_fn=seed_worker,generator=g,)
videomaxlen=200


frame,sentence,POShot,labellen,framelen,caption= training_data[0]

n_position_sentence=sentence.size()[0]+5
n_position_frame=(videomaxlen+5)
n_position=max(n_position_frame,n_position_sentence)
filename='checkpoint.pth.tar'
D_model_s=300
POS_enumerator = dataload.POS_enumerator
PosDim=len(POS_enumerator)
D_model_d=263+1

DecayRate=0.9975
END_THRE=0.5

lambda_e=0.04
lambda_emb=4
lambda_recon=10

BISHO=1.0e-10



cuda=torch.cuda.is_available()

if cuda:
    device = torch.device(f'cuda:{gpu_ids[0]}')
else:
    device = torch.device('cpu')


def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


class IndexPositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(IndexPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x,index):
        return torch.cat((x ,((self.pos_table[:,index:index+1, :x.size(2)]).clone().detach()).repeat(x.size()[0],1,1)),2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return torch.cat((x ,(self.pos_table[:,:x.size(1), :x.size(2)].clone().detach()).repeat(x.size()[0],1,1)),2)


def get_pad_mask(seq, seq_len):
    pad=torch.zeros(seq.size()[0],seq.size()[1])
    for i in range(seq.size()[0]):
        for j in range(seq_len[i]):
            pad[i][j]=1
    if cuda:
        return pad.bool().unsqueeze(-2).detach().to(seq.device)
    else:
        return pad.bool().unsqueeze(-2).detach()


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()[0],seq.size()[1]
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    if cuda:
        return subsequent_mask.detach().to(seq.device)
    else:
        return subsequent_mask.detach()

class Data_to_Emb(nn.Module):

    def __init__(
            self,d_model_d, d_model_f,d_inner, dropout=0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lrelu1=nn.LeakyReLU()
        self.lstm = nn.LSTM(d_model_d-1, d_model_d-1,batch_first=True,bidirectional=True)
        self.layer1 = nn.Linear(d_model_d+d_model_d-2, d_inner)
        self.layer2 = nn.Linear(d_inner,d_model_f-1)
        self.d_model_d=d_model_d-1

    def init_hidden(self,device,batch_size):

        return (torch.zeros(2, batch_size, self.d_model_d).to(device),torch.zeros(2, batch_size, self.d_model_d).to(device))

    def forward(self, x, slf_attn_mask):
        x2=x[:,:,:-1]
        self.hidden=self.init_hidden(x2.device,x2.size(0))
        textlen=x2.size(1)
        x2=pack_padded_sequence(x2, slf_attn_mask.cpu(), enforce_sorted=False,batch_first=True)
        lstm_out, self.hidden = self.lstm(x2, self.hidden)
        lstm_out2,len_batch=pad_packed_sequence(lstm_out,batch_first=True,total_length=textlen)
        x2 = self.layer2(self.lrelu1(self.layer1(lstm_out2)))
        x2 = self.dropout(x2)
        orig_emb=torch.zeros(x2.size(0),x2.size(1),x2.size(2)+1).to(x2.device)
        orig_emb[:,:,:-1]=x2
        orig_emb[:,:,-1:]=x[:,:,-1:]

        return orig_emb



class Emb_to_Data(nn.Module):

    def __init__(
            self,d_model_d, d_model_f,d_inner,n_position, dropout=0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lrelu1=nn.LeakyReLU()
        self.lrelu2=nn.LeakyReLU()
        self.position_enc = IndexPositionalEncoding(d_model_d, n_position)
        self.position_enc_train = PositionalEncoding(d_model_d, n_position)
        self.lstm = nn.LSTM(d_model_d+d_model_d, d_model_d+d_model_d,batch_first=True,bidirectional=False)
        self.layer1 = nn.Linear(d_model_d+d_model_d+d_model_f-1, d_inner)
        self.layer2 = nn.Linear(d_inner,d_model_d)
        self.layerm = nn.Linear(d_inner,d_inner)
        self.d_model_d=d_model_d

        self.DPsoftminLoss=DPsoftminLoss(D_model_d-1)
    def init_hidden(self,device,batch_size):

        return (torch.zeros(1, batch_size, self.d_model_d*2).to(device),torch.zeros(1, batch_size, self.d_model_d*2).to(device))

    def forward(self, emb_seq, slf_attn_mask,frame,if_training,if_rec2=False):
        xe=emb_seq[:,:,-1:]
        xe=torch.where(xe==float('inf'),torch.zeros(xe.size()).to(xe.device),xe)
        self.hidden=self.init_hidden(emb_seq.device,emb_seq.size(0))
        trg_seq=torch.zeros(xe.size(0),xe.size(1)+1,self.d_model_d).to(xe.device)
        trg_seq_mask=torch.zeros(xe.size(0),xe.size(1),self.d_model_d).to(xe.device)
        if if_rec2:
            x_emb=emb_seq.clone()
            x_emb[:,:,:-1]=self.dropout(emb_seq[:,:,:-1])
            self.hidden=self.init_hidden(emb_seq.device,emb_seq.size(0))
            for index in range(xe.size(1)):
                trg_seq2=self.position_enc(trg_seq[:,index:index+1],index)
                dec_output = self.dropout(trg_seq2)
                lstm_out, self.hidden = self.lstm(dec_output, self.hidden)
                dec_output2=torch.cat((lstm_out,x_emb[:,index:index+1,:-1]),-1)

                x = self.lrelu1(self.layer1(dec_output2))
                x = self.layer2(x + self.lrelu2(self.layerm(x)))

                trg_seq[:,index+1:index+2]=x
                trg_seq[:,index+1:index+2,-1:]=xe[:,index:index+1]

            return trg_seq[:,1:]

        if if_training:
            trg_seq=torch.zeros(xe.size(0),xe.size(1),self.d_model_d).to(xe.device)
            frame=torch.where(frame==float('inf'),trg_seq_mask,frame)

            trg_seq[:,1:,:]=frame[:,:-1,:]
            trg_seq2=self.position_enc_train(trg_seq)

            self.hidden=self.init_hidden(emb_seq.device,emb_seq.size(0))
            dec_output = self.dropout(trg_seq2)
            x_emb=emb_seq.clone()
            x_emb[:,:,:-1]=self.dropout(emb_seq[:,:,:-1])

            lstm_out, self.hidden = self.lstm(dec_output, self.hidden)

            dec_output2=torch.cat((lstm_out.unsqueeze(2).repeat(1,1,emb_seq.size(1),1),x_emb[:,:,:-1].unsqueeze(1).repeat(1,lstm_out.size(1),1,1)),-1)

            x = self.lrelu1(self.layer1(dec_output2))
            x = self.layer2(x + self.lrelu2(self.layerm(x)))

            ans,switchpos=self.DPsoftminLoss(x,frame,slf_attn_mask)

            get_index=(torch.cummax((torch.arange(emb_seq.size(1)).unsqueeze(0).to(emb_seq.device))*switchpos,1)[0]).unsqueeze(2).repeat(1,1,emb_seq.size(2))
            inputemb=torch.gather(emb_seq.to('cpu'),1,get_index.to('cpu')).to(emb_seq.device)
            inputemb[:,:,-1:]=self.dropout(inputemb[:,:,-1:])
            trg_seq=torch.zeros(xe.size(0),xe.size(1)+1,self.d_model_d).to(xe.device)

            self.hidden=self.init_hidden(emb_seq.device,emb_seq.size(0))
            for index in range(xe.size(1)):
                trg_seq2=self.position_enc(trg_seq[:,index:index+1],index)
                dec_output = self.dropout(trg_seq2)
                lstm_out, self.hidden = self.lstm(dec_output, self.hidden)

                dec_output2=torch.cat((lstm_out,inputemb[:,index:index+1,:-1]),-1)


                x = self.lrelu1(self.layer1(dec_output2))
                x = self.layer2(x + self.lrelu2(self.layerm(x)))

                trg_seq[:,index+1:index+2]=x
                trg_seq[:,index+1:index+2,-1:]=xe[:,index:index+1]

            synt_frame=trg_seq[:,1:].clone()

            rec=0

            Recon=ReconLoss.ReconLoss()
            loss=Recon(frame[:,:,0:D_model_d-1],synt_frame[:,:frame.size(1),0:D_model_d-1],slf_attn_mask,slf_attn_mask)
            rec=torch.sum(loss)

            return ans,switchpos,x,rec

        for index in range(xe.size(1)):
            trg_seq2=self.position_enc(trg_seq[:,index:index+1],index)
            dec_output = self.dropout(trg_seq2)
            lstm_out, self.hidden = self.lstm(dec_output, self.hidden)
            x_emb=emb_seq.clone()
            x_emb[:,:,:-1]=self.dropout(emb_seq[:,:,:-1])

            dec_output2=torch.cat((lstm_out,x_emb[:,index:index+1,:-1]),-1)

            x = self.lrelu1(self.layer1(dec_output2))
            x = self.layer2(x + self.lrelu2(self.layerm(x)))

            trg_seq[:,index+1:index+2]=x
            trg_seq[:,index+1:index+2,-1:]=xe[:,index:index+1]

        return trg_seq[:,1:]

class AttnEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,d_word_vec, n_layers, n_head, d_k, d_v,
            d_model_i, d_inner,dropout=0.1, n_position=n_position):

        super().__init__()

        self.pos_emb = nn.Linear(PosDim, d_model_i)
        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model_i, n_position)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model_i, d_inner, n_head, d_k, d_v,n_position,BATCH_SIZE, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, POShot,src_mask):

        pos_emb=self.pos_emb(POShot)
        src_seq=src_seq+pos_emb

        # -- Forward

        enc_output = self.position_enc(src_seq)
        enc_output = self.dropout(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=src_mask)

        enc_output=enc_output+torch.cat((src_seq, (torch.zeros(src_seq.size())).to(src_seq.device)), 2)

        return enc_output


class Emb_distribute(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512,n_layers=1):
        super(Emb_distribute, self).__init__()
        self.layer1=nn.Linear(hidden_size, hidden_size)
        self.LR1=nn.LeakyReLU(0.2, inplace=True)
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)



    def forward(self, inputs):
        x_in = self.emb(inputs)
        hidden = self.LR1(self.layer1(x_in))
        mu = self.mu_net(hidden)
        logvar = self.logvar_net(hidden)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class TextVAEDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, n_layers=1):
        super(TextVAEDecoder, self).__init__()
        self.sig=nn.Sigmoid()
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        h_in = self.emb(inputs)
        pose_pred = self.output(h_in)
        pose_pred[:,:,-1:]=self.sig(pose_pred[:,:,-1:])
        return pose_pred


class AttnDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,d_word_vec, n_layers, n_head, d_k, d_v,
            d_model_i,d_model_o,d_frame, d_inner, videomaxlen, n_position=200, dropout=0.1,d_z=512):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.videomaxlen=videomaxlen

        self.seq_pri = Emb_distribute(d_model_o+d_model_o+d_model_o+d_model_o,d_z)
        self.seq_post = Emb_distribute(d_model_o+d_model_o+d_model_o+d_model_o+d_model_o,d_z)
        self.seq_dec =TextVAEDecoder(d_model_o+d_model_o+d_model_o+d_model_o+d_z,d_frame)

        self.trg_word_prj = nn.Linear(d_model_o+d_model_o, d_frame, bias=False)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model_i,d_model_o, d_inner, n_head, d_k, d_v,videomaxlen,BATCH_SIZE,n_position,dropout=dropout)
            for _ in range(n_layers)])

        self.position_enc = IndexPositionalEncoding(d_model_o, n_position)
        self.layernorm = nn.LayerNorm(d_frame-1, eps=1e-6)

    def kl_criterion(self,mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
                2 * torch.exp(logvar2)) - 1 / 2
        return kld

    def forward(self,inputemb, trg_seq, trg_mask, enc_output, src_mask, if_eval=False):


        mus_pri = []
        logvars_pri = []
        mus_post = []
        logvars_post = []
        fake_mov_batch = []
        temp_zeros=torch.zeros(inputemb[:, 0:1].size()).to(inputemb.device)

        for index in range(self.videomaxlen):
            trg_seq2=self.position_enc(trg_seq[:,index:index+1],index)
            dec_output = self.dropout(trg_seq2)
            if index<inputemb.size(1):
                mov_tgt = inputemb[:, index:index+1]
            else:
                mov_tgt = torch.zeros(inputemb[:, 0:1].size()).to(inputemb.device)

            mov_tgt=torch.where(mov_tgt==float('inf'),temp_zeros,mov_tgt)

            for dec_layer in self.layer_stack:
                dec_output, dec_enc_attn = dec_layer(
                    dec_output, enc_output,index, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

            att_vec=dec_output

            pos_in = torch.cat([trg_seq2, mov_tgt, att_vec], dim=-1)
            pri_in = torch.cat([trg_seq2, att_vec], dim=-1)

            z_pos, mu_pos, logvar_pos= self.seq_post(pos_in)

            '''Prior'''
            z_pri, mu_pri, logvar_pri= self.seq_pri(pri_in)

            if (not self.training) and (not if_eval):
                dec_in = torch.cat([trg_seq2, att_vec, z_pri], dim=-1)
            else:
                dec_in = torch.cat([trg_seq2, att_vec, z_pos], dim=-1)
            fake_mov = self.seq_dec(dec_in)

            mus_post.append(mu_pos)
            logvars_post.append(logvar_pos)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)
            fake_mov_batch.append(fake_mov.unsqueeze(1))

            trg_seq[:,index+1:index+2]=fake_mov

        self.fake_movements = torch.cat(fake_mov_batch, dim=1)
        self.mus_post = torch.cat(mus_post, dim=1)
        self.mus_pri = torch.cat(mus_pri, dim=1)
        self.logvars_post = torch.cat(logvars_post, dim=1)
        self.logvars_pri = torch.cat(logvars_pri, dim=1)
        return trg_seq, self.kl_criterion(self.mus_post, self.logvars_post, self.mus_pri, self.logvars_pri)

class Attn(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            d_word_vec=D_model_s, d_model_i=D_model_s,d_model_o=D_model_f, d_inner=64,
            n_layers_enc=1,n_layers_dec=1, n_head=8, d_k=32, d_v=32, dropout=0.1, n_position=n_position,d_frame=D_model_f
            ):

        super().__init__()

        self.attn_encoder = AttnEncoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model_i=d_model_i, d_inner=d_inner,
            n_layers=n_layers_enc, n_head=n_head, d_k=d_k, d_v=d_v,
             dropout=dropout)

        self.attn_decoder = AttnDecoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model_i=d_model_i,d_model_o=d_model_o, d_inner=d_inner,d_frame=d_frame,
            n_layers=n_layers_dec, n_head=n_head, d_k=d_k, d_v=d_v,
             dropout=dropout,videomaxlen=videomaxlen)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model_i == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, inputemb,src_seq, POShot,trg_seq,labellen,framelen,epoch,if_eval=False):

        src_mask = labellen
        src_mask2 = get_pad_mask(src_seq, labellen)
        trg_mask = get_subsequent_mask(trg_seq)

        enc_output= self.attn_encoder(src_seq,POShot, src_mask)
        dec_output, klloss= self.attn_decoder(inputemb,trg_seq, trg_mask, enc_output, src_mask2,if_eval=if_eval)

        fake_video=dec_output[:,1:,:]

        ifend=fake_video[:,:,-1]
        endframelist=[]
        for i in range(ifend.size(0)):
            tl=torch.where(ifend[i]>END_THRE)[0]
            if(len(tl)==0):
                endframenum=(videomaxlen-1)
            else:
                endframenum=(int(tl[0])+1)
            endframelist.append(endframenum)

        return fake_video,endframelist,klloss

class TextModule(nn.Module):
    def __init__(
            self,
            d_word_vec=D_model_s, d_model_i=D_model_s,d_model_o=D_model_f, d_inner=64,
            n_layers_enc=1,n_layers_dec=1, n_head=8, d_k=32, d_v=32, dropout=0.1, n_position=n_position,d_frame=D_model_f
            ):
        super().__init__()

        self.attn=Attn(d_word_vec, d_model_i,d_model_o, d_inner,n_layers_enc,n_layers_dec, n_head, d_k, d_v, dropout, n_position,d_frame)

    def forward(self,inputemb, src_seq,POShot, trg_seq,labellen,framelen,frame,epoch,if_eval=False):
        fake_video, endframelist,klloss=self.attn(inputemb,src_seq,POShot, trg_seq,labellen,framelen,epoch,if_eval=if_eval)
        return fake_video,endframelist,klloss

class DataModule(nn.Module):
    def __init__(
            self,
            d_word_vec=D_model_s, d_model_i=D_model_s,d_model_o=D_model_f, d_inner=64,d_innerEmbtoData=128,
            n_layers_enc=1,n_layers_dec=1, n_head=8, d_k=32, d_v=32, dropout=0.1, n_position=n_position,d_frame=D_model_f
            ):
        super().__init__()

        self.data_to_emb=Data_to_Emb(D_model_d, D_model_f,d_inner)
        self.emb_to_data=Emb_to_Data(D_model_d, D_model_f,d_innerEmbtoData,n_position)

    def forward(self, src_seq, trg_seq,labellen,framelen,frame,epoch):
        orig_emb=self.data_to_emb(frame,framelen)
        ans,switchpos,output,rec=self.emb_to_data(orig_emb,framelen,frame,True)
        get_index=(torch.cummax((torch.arange(orig_emb.size(1)).unsqueeze(0).to(orig_emb.device))*switchpos,1)[0]).unsqueeze(2).repeat(1,1,orig_emb.size(2))
        inputemb=torch.gather(orig_emb.to('cpu'),1,get_index.to('cpu')).to(orig_emb.device)
        inputemb[:,:,-1:]=orig_emb[:,:,-1:]
        output=self.emb_to_data(inputemb,framelen,frame,False)
        switchnum=torch.sum(torch.gather(torch.cumsum(switchpos.to("cpu"),-1),1,(framelen-1).to('cpu').unsqueeze(1))).to(output.device)

        return output, ans,inputemb,orig_emb,rec,switchnum



class ModelNetwork(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            d_word_vec=D_model_s, d_model_i=D_model_s,d_model_o=D_model_f, d_inner=64,d_innerEmbtoData=D_innerEmbtoData,
            n_layers_enc=1,n_layers_dec=1, n_head=8, d_k=32, d_v=32, dropout=0.1, n_position=n_position,d_frame=D_model_f
            ):

        super().__init__()


        self.dataM=DataModule(d_word_vec, d_model_i,d_model_o, d_inner,d_innerEmbtoData,
        n_layers_enc,n_layers_dec, n_head, d_k, d_v, dropout, n_position,d_frame
        )

        self.textM=TextModule(d_word_vec, d_model_i,d_model_o, d_inner,
        n_layers_enc,n_layers_dec, n_head, d_k, d_v, dropout, n_position,d_frame
        )


    def forward(self, src_seq, POShot,trg_seq,labellen,framelen,frame,epoch,if_eval=False,if_rec2=False):
        output, ans,inputemb,orig_emb,rec,switchnum=self.dataM(src_seq, trg_seq,labellen,framelen,frame,epoch)
        fake_video,endframelist,klloss=self.textM(inputemb,src_seq,POShot, trg_seq,labellen,framelen,frame,epoch,if_eval=if_eval)

        output2=model.dataM.emb_to_data(fake_video[:,:frame.size(1)],framelen,frame,False,if_rec2=True)

        alignloss=0
        if self.training or if_eval:
            val=0
            val2=0
            templ=inputemb.size(1)
            loss_shape=torch.sum(torch.gather(torch.cumsum(torch.mean((inputemb[:,:,:-1]-fake_video[:,:templ,:-1])*(inputemb[:,:,:-1]-fake_video[:,:templ,:-1]),-1),1).to('cpu'),1,(framelen-1).to('cpu').unsqueeze(1)).squeeze(1)).to(fake_video.device)
            val=val+loss_shape
            loss_shape2=torch.sum(torch.gather(torch.cumsum(torch.mean((orig_emb[:,:,:-1]-fake_video[:,:templ,:-1])*(orig_emb[:,:,:-1]-fake_video[:,:templ,:-1]),-1),1).to('cpu'),1,(framelen-1).to('cpu').unsqueeze(1)).squeeze(1)).to(fake_video.device)

            val2=val2+loss_shape2

            frame_roll=torch.roll(inputemb, 1, 0)
            framelen_roll=torch.roll(framelen, 1, 0)

            val_n2 = torch.sum(torch.gather(torch.cumsum(torch.mean((inputemb[:,:,:-1]-frame_roll[:,:,:-1])*(inputemb[:,:,:-1]-frame_roll[:,:,:-1]),-1),1).to('cpu'),1,(torch.min(framelen,framelen_roll)-1).to('cpu').unsqueeze(1)).squeeze(1)).to(inputemb.device)
            val_n3 = torch.sum(torch.gather(torch.cumsum(torch.mean((fake_video[:,:templ,:-1]-frame_roll[:,:,:-1])*(fake_video[:,:templ,:-1]-frame_roll[:,:,:-1]),-1),1).to('cpu'),1,(framelen_roll-1).to('cpu').unsqueeze(1)).squeeze(1)).to(inputemb.device)
            klloss_sum=torch.sum(torch.gather(torch.cumsum(torch.sum(klloss,-1),1).to('cpu'),1,(framelen-1).to('cpu').unsqueeze(1))).to(inputemb.device)
            alignloss=(val2+val)/(val_n2+val_n3+BISHO)
            saveval=torch.sum(val).data
            saveval_n=0
            val=val+(alignloss-alignloss.detach())*lambda_a

        else:
            val=0
            valn=0
            klloss_sum=0
            templ=inputemb.size(1)
            val=torch.sum(torch.gather(torch.cumsum(torch.mean((inputemb[:,:,:-1]-fake_video[:,:templ,:-1])*(inputemb[:,:,:-1]-fake_video[:,:templ,:-1]),-1),1).to('cpu'),1,(framelen-1).to('cpu').unsqueeze(1)).squeeze(1)).to(fake_video.device)
            with torch.no_grad():
                frame_roll=torch.roll(inputemb, 1, 0)
                framelen_roll=torch.roll(framelen, 1, 0)
                val2=torch.sum(torch.gather(torch.cumsum(torch.mean((frame_roll[:,:,:-1]-fake_video[:,:templ,:-1])*(frame_roll[:,:,:-1]-fake_video[:,:templ,:-1]),-1),1).to('cpu'),1,(framelen_roll-1).to('cpu').unsqueeze(1)).squeeze(1)).to(fake_video.device)
            saveval=torch.sum(val).data
            saveval_n=torch.sum(val2).data
        tmp_emb=(inputemb[:,:,:-1].detach()-orig_emb[:,:,:-1])
        dif_emb=torch.sum(torch.gather(torch.cumsum(torch.mean((tmp_emb)*(tmp_emb),-1),1).to('cpu'),1,(framelen-1).to('cpu').unsqueeze(1)).squeeze(1)).to(tmp_emb.device)
        val=val+ lambda_proc*(dif_emb-dif_emb.detach())
        G_endloss=0
        ifend=fake_video[:,:,-1]
        for i in range(fake_video.size(0)):
            G_endloss+=-1*torch.mean(torch.sum(torch.log(ifend[i][(framelen[i]-1):]*(1-2*BISHO)+BISHO))+torch.sum(torch.log((1-ifend[i][:framelen[i]-1])*(1-2*BISHO)+BISHO)))

        if if_eval:
            return output, output2,ifend,val,G_endloss,saveval,alignloss, ans,saveval_n,dif_emb,rec,klloss_sum, switchnum
        return output, output2,ifend,val,G_endloss,saveval,alignloss, ans,saveval_n,dif_emb,rec,klloss_sum

if __name__ == '__main__':
#with detect_anomaly():

    model=ModelNetwork()
    Recon=ReconLoss.ReconLoss()

    Gdata_optimizer = optim.Adam(model.dataM.parameters(), lr=0.0004, betas=(0.5, 0.999))
    Gtext_optimizer = optim.Adam(model.textM.parameters(), lr=0.0004, betas=(0.5, 0.999))
    schedulerD=torch.optim.lr_scheduler.ExponentialLR(optimizer=Gdata_optimizer, gamma=DecayRate)
    schedulerT=torch.optim.lr_scheduler.ExponentialLR(optimizer=Gtext_optimizer, gamma=DecayRate)

    G_evalloss=None
    epoch = 0
    loss_count=0
    if cuda:
        model.to(device)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)

        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        Gdata_optimizer.load_state_dict(checkpoint['Doptimizer'])
        Gtext_optimizer.load_state_dict(checkpoint['Toptimizer'])
        torch.set_rng_state(checkpoint['random'])
        loss_count=checkpoint['loss_count']

        G_evalloss = checkpoint['evalloss'].to(device)
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        schedulerT.load_state_dict(checkpoint['schedulerT'])


        for state in Gtext_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for state in Gdata_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))




    while(epoch<500):

        if loss_count>30 and epoch>200:
            break
        model.train()

        G_endsum=torch.zeros(1).to(device)
        G_reconsum=torch.zeros(1).to(device)
        G_attnsum=0
        G_difsum=0
        dif_embsum=0

        savevalsum=0
        saverec2=0
        saverec=0
        save_reconloss=0
        alignlosssum=0
        savekl=0
        G_running_loss=0
        model.train()
        G_loss=0
        iter=0
        print("epoch=",epoch)
        for frame,sentence,POShot,labellen,framelen,_ in train_loader:
            iter+=1
            if cuda:
                frame,sentence,POShot,labellen,framelen=frame.float().to(device),sentence.float().to(device),POShot.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence,POShot=frame.float(),sentence.float(),POShot.float()

            Gtext_optimizer.zero_grad()
            Gdata_optimizer.zero_grad()

            input=torch.zeros(frame.size(0),videomaxlen+1,D_model_f)

            if cuda:
                input=input.to(device)

            fake_video, output2,ifend,val,G_endloss,saveval,alignloss,G_reconloss,saveval_n,dif_emb,rec,klloss_sum=model(sentence,POShot,input,labellen,framelen,frame,epoch)

            G_difloss=0
            G_attnloss=0
            G_difloss+=torch.sum(val)

            rec2=torch.sum(Recon(frame[:,:,0:(D_model_d-1)],output2[:,:frame.size(1),0:(D_model_d-1)],framelen,framelen))

            G_loss=lambda_e*G_endloss+lambda_emb*G_difloss+lambda_recon*(G_reconloss+lambda_r*rec)+lambda_KL*klloss_sum
            G_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),CLIP_MODEL)
            Gdata_optimizer.step()
            Gtext_optimizer.step()

            G_difsum+=G_difloss.data
            if G_endloss!=0:
                G_endsum+=G_endloss.data

            if G_reconloss!=0:
                G_reconsum+=G_reconloss.data

            savevalsum+=saveval.data
            alignlosssum+=alignloss.data
            dif_embsum+=torch.sum(dif_emb).data
            saverec+=rec.data
            saverec2+=rec2.data
            savekl+=klloss_sum.data

        writer.add_scalar('savevalsum', savevalsum, epoch)
        writer.add_scalar('alignlosssum', alignlosssum, epoch)
        writer.add_scalar("KL",lambda_KL*savekl,epoch)
        writer.add_scalar('end',lambda_e*G_endsum, epoch)
        writer.add_scalar('recon',lambda_recon*G_reconsum, epoch)
        writer.add_scalar('rec',lambda_recon*saverec*lambda_r, epoch)

        writer.add_scalar('difloss',lambda_emb*G_difsum, epoch)
        writer.add_scalar("dif_emb",dif_embsum,epoch)

        G_running_loss = G_loss.data

        print("G_running_loss=",G_running_loss)

        with torch.no_grad():
            model.eval()
            G_temploss=0
            G_difsum=0
            G_endsum=0
            savekl=0
            saverec=0
            saverec2=0

            G_reconloss=0
            sum_val=0
            sum_valn=0
            sum_switchnum=0
            for frame,sentence,POShot,labellen,framelen,_ in eval_loader:
                if cuda:
                    frame,sentence,POShot,labellen,framelen=frame.float().to(device),sentence.float().to(device),POShot.float().to(device),labellen.to(device),framelen.to(device)
                else:
                    frame,sentence,POShot=frame.float(),sentence.float(),POShot.float()


                Gtext_optimizer.zero_grad()
                Gdata_optimizer.zero_grad()

                input=torch.zeros(frame.size(0),videomaxlen+1,D_model_f)
                if cuda:
                    input=input.to(device)

                with torch.no_grad():
                    fake_video,output2, ifend,val,G_endloss,saveval,alignloss,G_reconloss,saveval_n,dif_emb,rec,klloss_sum,switchnum=model(sentence,POShot,input,labellen,framelen,frame,epoch,if_eval=True)
                    rec2=torch.sum(Recon(frame[:,:,0:(D_model_d-1)],output2[:,:frame.size(1),0:(D_model_d-1)],framelen,framelen))

                G_difloss=0
                G_difloss+=torch.sum(val)
                G_loss=lambda_recon*(lambda_r*(rec2+P_PENAL*switchnum))+lambda_KL*klloss_sum
                G_temploss+=G_loss
                savekl+=klloss_sum
                saverec+=rec
                saverec2+=rec2
                G_difsum+=G_difloss
                G_endsum+=G_endloss
                sum_switchnum+=switchnum
            writer.add_scalar('evalloss',G_temploss, epoch)
            writer.add_scalar("evalrec",lambda_recon*lambda_r*saverec,epoch)
            writer.add_scalar("evalrec2",lambda_recon*lambda_r*saverec2,epoch)
            writer.add_scalar("eval_kkl",lambda_KL*savekl,epoch)
            writer.add_scalar("evaldifloss",lambda_emb*G_difsum,epoch)
            writer.add_scalar("evalswitchnum",lambda_recon*lambda_r*P_PENAL*sum_switchnum,epoch)

            if (G_evalloss==None or G_temploss<G_evalloss):
                torch.save(model.state_dict(), "text20_res_model.pt")
                print(G_evalloss,G_temploss)
                G_evalloss=G_temploss
                loss_count=0

            else:
                loss_count+=1

        epoch+=1

        schedulerD.step()
        schedulerT.step()

        state = {'epoch': epoch, 'state_dict': model.state_dict(),'Toptimizer': Gtext_optimizer.state_dict(),'Doptimizer': Gdata_optimizer.state_dict(), 'random': torch.get_rng_state(), 'evalloss': G_evalloss.to('cpu'),'schedulerT': schedulerT.state_dict(),'schedulerD': schedulerD.state_dict(),'loss_count':loss_count,}
        torch.save(state, filename)

    writer.close()

    model.load_state_dict(torch.load("text20_res_model.pt"))

    model.dataM.emb_to_data.position_enc.pos_table=model.dataM.emb_to_data.position_enc._get_sinusoid_encoding_table(n_position*2,D_model_d).to(device)
    model.dataM.emb_to_data.position_enc_train.pos_table=model.dataM.emb_to_data.position_enc_train._get_sinusoid_encoding_table(n_position*2,D_model_d).to(device)
    model.textM.attn.attn_encoder.position_enc.pos_table=model.textM.attn.attn_encoder.position_enc._get_sinusoid_encoding_table(n_position*2,D_model_s).to(device)
    model.textM.attn.attn_decoder.position_enc.pos_table=model.textM.attn.attn_decoder.position_enc._get_sinusoid_encoding_table(n_position*2,D_model_f).to(device)


    with torch.no_grad():
        model.eval()
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE, num_workers=2, shuffle=False,worker_init_fn=seed_worker,generator=g,)
        i=0
        for frame,sentence,POShot,labellen,framelen,caption in test_loader:
            i+=1
            if cuda:

                frame,sentence,POShot,labellen,framelen=frame.float().to(device),sentence.float().to(device),POShot.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence,POShot=frame.float(),sentence.float(),POShot.float()


            input=torch.zeros(frame.size(0),videomaxlen+1,D_model_f)
            if cuda:
                input=input.to(device)
            output, ans,inputemb,orig_emb,rec,switchnum=model.dataM(sentence,input,labellen,framelen,frame,epoch)

            inputemb2, endframelist,klloss=model.textM.attn(torch.zeros(inputemb.size()).to(inputemb.device),sentence,POShot,input,labellen,torch.zeros(framelen.size()).long(),epoch)
            endframelist=[]
            for en in range(frame.size(0)):
                tl=torch.where(inputemb2[en,:,-1]>END_THRE)[0]
                if(len(tl)==0):
                    endframenum=(len(inputemb2[0]))
                else:
                    endframenum=(int(tl[0])+1)

                endframelist.append(endframenum)
            endframelist=torch.LongTensor(endframelist)
            fake_video=model.dataM.emb_to_data(inputemb2,endframelist,None,False)

            ifend=fake_video[:,:,-1]

            interpolate_emb=inputemb.unsqueeze(-1).repeat(1,1,2,1).view(inputemb.size(0),inputemb.size(1)*2,inputemb.size(2))
            interp_len=framelen*2-1
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            interpolate_emb2=inputemb2.unsqueeze(-1).repeat(1,1,2,1).view(inputemb2.size(0),inputemb2.size(1)*2,inputemb2.size(2))
            interp_len=endframelist*2-1
            interp2=model.dataM.emb_to_data(interpolate_emb2,interp_len,None,False)
            regt=model.dataM.emb_to_data(inputemb,framelen,None,False)

            for j in range(frame.size(0)):
                tl=torch.where(fake_video[j,:,-1]>END_THRE)[0]
                if(len(tl)==0):
                    endframenum=(videomaxlen+2)
                else:
                    endframenum=(int(tl[0])+1)
                path_name='text20_generatadvideo/{:07d}/'.format((num_train+i)*BATCH_SIZE+j)
                os.makedirs(path_name, exist_ok=True)
                path_txt=path_name+'story.txt'

                with open(path_txt, mode='w') as f:
                    f.write(caption[j])
                np.savez(path_name+'data', re_gt=regt[j][:framelen[j]].to('cpu').detach().numpy(),people=fake_video[j].to('cpu').detach().numpy().copy(),gt=frame[j].to('cpu').detach().numpy().copy(),\
                gen_framelen=endframenum,gt_framelen=framelen[j].to('cpu').detach().numpy().copy(),interpolate=interp[j].to('cpu').detach().numpy().copy(),interpolate2=interp2[j].to('cpu').detach().numpy().copy(),\
                sentence=sentence[j].to('cpu').detach().numpy().copy(),POShot=POShot[j].to('cpu').detach().numpy().copy())
