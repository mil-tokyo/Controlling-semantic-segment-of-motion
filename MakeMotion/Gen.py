''' 2Define the Transformer model '''
seed=1
D_innerEmbtoData=512
P_KL=0.01
import os
os.environ["OMP_NUM_THREADS"] = "1"
expW=1
D_model_f=32##temp! this is the dim of meaning vector.
P_VAL=200
RRario=10
RRario2=0
alpha_frame=0.5
alpha=0.3
gamma = 0.01
P_emb=1
from loss.dilate_loss import dilate_loss
from loss_dilateframe.dilate_loss import dilate_loss as dilate_frame
from tslearn.metrics import dtw, dtw_path
#add neighbor elem loss
import torch
from DPsoftminLoss import DPsoftminLoss
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
num_train2=8000
import sys
gpu_ids= [(int)(sys.argv[1])]
BATCH_SIZE=50
subtract_ifmatched=0.1
bonelist=['mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head','mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm','mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm','mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg']


import torch
import random
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)
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
eval_train=1000
DATA_NUM=10000
num_test=1000
thre_attn=0.1
align_thre=0.5
AlignOverWeight=1
NeighborLossWeight=0.0


videomaxlen=200

'''
test_loader = torch.utils.data.DataLoader(dataset=test_data[num_train:],
                         batch_size=BATCH_SIZE, shuffle=False)
'''
def neighbor_elem_loss(frame):
    output_loss=torch.sqrt(0.25+torch.sum(torch.abs(frame),-1))
    return torch.sum(output_loss)


filename='checkpoint.pth.tar'
D_model_s=300
POS_enumerator = dataload.POS_enumerator
PosDim=len(POS_enumerator)
D_model_d=263+1

DecayRate=0.9975
END_THRE=0.5

P_DIFFRAME=0.04
P_DIFFRAME2=4
P_DIFFRAME3=2
P_DIFFRAME4=1
P_DIFFRAMERECON=1
P_DIFFRAME_attn=0
BISHO=1.0e-10
W=700
H=400


cuda=torch.cuda.is_available()

if cuda:
    device = torch.device(f'cuda:{gpu_ids[0]}')
else:
    device = torch.device('cpu')

#cuda=False


__author__ = "Yu-Hsiang Huang"




if __name__ == '__main__':
#with detect_anomaly():


    gpu_id=(int)(sys.argv[1])
    cuda=torch.cuda.is_available()
    if cuda:
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    mod = sys.argv[2]
    sys.path.append("../"+mod)
    import Models as models

    import Testdataload
    import torch.nn.functional as F
    import torch.optim as optim
    CLIP_MODEL=100
    num_train=8000
    eval_train=1000
    DATA_NUM=10000
    num_test=1000

    from Models import ModelNetwork,training_data,eval_data,train_loader,eval_loader,videomaxlen,BATCH_SIZE,D_model_f,D_model_s,D_model_d,END_THRE
    test_data = Testdataload.Text2MotionDataset_test("../../../dataset/HumanML3D/HumanML3D/test.txt")

    MAXLEN=(int)(videomaxlen*1.5)
    frame,sentence,POShot,labellen,framelen,caption= training_data[0]

    n_position_sentence=sentence.size()[0]+5
    textlen=sentence.size()[0]
    n_position_frame=(videomaxlen+5)
    n_position=max(n_position_frame,n_position_sentence)
    model=ModelNetwork()
    model.load_state_dict(torch.load("../"+mod+"/text20_res_model.pt",map_location=device))

    if cuda:
        model.to(device)

    model.dataM.emb_to_data.position_enc.pos_table=model.dataM.emb_to_data.position_enc._get_sinusoid_encoding_table(n_position*2,D_model_d).to(device)
    model.dataM.emb_to_data.position_enc_train.pos_table=model.dataM.emb_to_data.position_enc_train._get_sinusoid_encoding_table(n_position*2,D_model_d).to(device)
    model.textM.attn.attn_encoder.position_enc.pos_table=model.textM.attn.attn_encoder.position_enc._get_sinusoid_encoding_table(n_position*2,D_model_s).to(device)
    model.textM.attn.attn_decoder.position_enc.pos_table=model.textM.attn.attn_decoder.position_enc._get_sinusoid_encoding_table(n_position*2,D_model_f).to(device)




    with torch.no_grad():
        epoch=0
        if_firstphase=True
        model.eval()
        #D.eval()
        #for data in train_loader:
        test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE, num_workers=2, shuffle=False,worker_init_fn=seed_worker,generator=g,)
        i=0
        for frame,sentence,POShot,labellen,framelen,caption in test_loader:
            i+=1
            if cuda:

                frame,sentence,POShot,labellen,framelen=frame.float().to(device),sentence.float().to(device),POShot.float().to(device),labellen.to(device),framelen.to(device)
            else:
                frame,sentence,POShot=frame.float(),sentence.float(),POShot.float()




            input=torch.zeros(frame.size(0),videomaxlen+1,D_model_f)
            image_t=torch.Tensor([W,H]).detach()
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
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)#fake_video[:,:,26:]=self.sig(fake_video[:,:,26:])

            interpolate_emb2=inputemb2.unsqueeze(-1).repeat(1,1,2,1).view(inputemb2.size(0),inputemb2.size(1)*2,inputemb2.size(2))
            interp_len=endframelist*2-1
            interp2=model.dataM.emb_to_data(interpolate_emb2,interp_len,None,False)#fake_video[:,:,26:]=self.sig(fake_video[:,:,26:])

            regt=model.dataM.emb_to_data(inputemb,framelen,None,False)#fake_video[:,:,26:]=self.sig(fake_video[:,:,26:])

            for j in range(frame.size(0)):
                tl=torch.where(fake_video[j,:,-1]>END_THRE)[0]
                #print("tl=",tl)

                if(len(tl)==0):
                    endframenum=(videomaxlen+2)
                else:
                    endframenum=(int(tl[0])+1)
                path_name="../text"+mod+'/text20_generatadvideo/{:07d}/'.format((num_train+i)*BATCH_SIZE+j)
                os.makedirs(path_name, exist_ok=True)
                path_txt=path_name+'story.txt'

                with open(path_txt, mode='w') as f:
                    f.write(caption[j])
                np.savez(path_name+'data', re_gt=regt[j][:framelen[j]].to('cpu').detach().numpy(),people=fake_video[j].to('cpu').detach().numpy().copy(),gt=frame[j].to('cpu').detach().numpy().copy(),\
                gen_framelen=endframenum,gt_framelen=framelen[j].to('cpu').detach().numpy().copy(),interpolate=interp[j].to('cpu').detach().numpy().copy(),interpolate2=interp2[j].to('cpu').detach().numpy().copy(),\
                sentence=sentence[j].to('cpu').detach().numpy().copy(),POShot=POShot[j].to('cpu').detach().numpy().copy())
