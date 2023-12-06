''' 2Define the Transformer model '''
seed=1
D_innerEmbtoData=512
lambda_KL=0.05
import os
os.environ["OMP_NUM_THREADS"] = "1"
D_model_f=32##temp! this is the dim of meaning vector.
lambda_r=5
gamma = 0.01
lambda_proc=1
import torch
import dataload2
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
num_train2=8000
import sys
gpu_ids= [(int)(sys.argv[1])]
model_dir=sys.argv[2]
BATCH_SIZE=50
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
lambda_a=1


test_data = dataload2.Text2MotionDataset2("../../../dataset/HumanML3D/HumanML3D/test.txt")





filename='checkpoint.pth.tar'
D_model_s=300
POS_enumerator = dataload2.POS_enumerator
PosDim=len(POS_enumerator)
D_model_d=263+1

DecayRate=0.9975
END_THRE=0.5

lambda_e=0.04
lambda_emb=2
lambda_recon=10

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

    print("begin")


    sys.path.append("../"+model_dir)

    from Models import ModelNetwork,D_model_f,videomaxlen,END_THRE,n_position,D_model_d,D_model_s


    model=ModelNetwork()

    G_evalloss=None
    epoch = 0
    loss_count=0
    if cuda:
        model.to(device)
    model.load_state_dict(torch.load("../"+model_dir+"/text20_res_model.pt",map_location=device))

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
            save_edit=[]
            savepos_list=[]




            #0
            interpolate_emb=torch.cat((inputemb[3][0:55],inputemb[3][55:69].unsqueeze(-1).repeat(1,3,1).view(-1,inputemb.size(2)),inputemb[3][55:]),0).unsqueeze(0)
            interp_len=(framelen[3]+42).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])

            #1
            interpolate_emb=torch.cat((inputemb[3][0:20],inputemb[3][20:34].unsqueeze(-1).repeat(1,3,1).view(-1,inputemb.size(2)),inputemb[3][20:]),0).unsqueeze(0)
            interp_len=(framelen[3]+42).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])

            #2
            interpolate_emb=torch.cat((inputemb[3][0:146],inputemb[8][76:87].unsqueeze(-1).repeat(1,3,1).view(-1,inputemb.size(2)),inputemb[3][146:]),0).unsqueeze(0)
            interp_len=(framelen[3]+33).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    if ii==178:
                        save_pos.append(ii+0.2)
                    else:
                        save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])










            #14
            interpolate_emb=inputemb[8][80:81].repeat(67,1).unsqueeze(0)
            interp_len=(torch.LongTensor([67]).to(framelen.device)).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])






            a=0.25
            interpolate_emb=((a*inputemb[3][55:56]+(1-a)*inputemb[3][20:21])).repeat(67,1).unsqueeze(0)
            interp_len=(torch.LongTensor([67]).to(framelen.device)).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])


            a=0.5
            interpolate_emb=((a*inputemb[3][55:56]+(1-a)*inputemb[3][20:21])).repeat(67,1).unsqueeze(0)
            interp_len=(torch.LongTensor([67]).to(framelen.device)).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])


            a=0.75
            interpolate_emb=((a*inputemb[3][55:56]+(1-a)*inputemb[3][20:21])).repeat(67,1).unsqueeze(0)
            interp_len=(torch.LongTensor([67]).to(framelen.device)).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])


            a=0
            interpolate_emb=((a*inputemb[3][55:56]+(1-a)*inputemb[3][20:21])).repeat(67,1).unsqueeze(0)
            interp_len=(torch.LongTensor([67]).to(framelen.device)).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                    save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])


            a=0
            interpolate_emb=((a*inputemb[3][20:21]+(1-a)*inputemb[3][55:56])).repeat(67,1).unsqueeze(0)
            interp_len=(torch.LongTensor([67]).to(framelen.device)).unsqueeze(0)
            interp=model.dataM.emb_to_data(interpolate_emb,interp_len,None,False)
            save_pos=[]
            for ii in range((int)(interp_len)-1):
                if not (interpolate_emb[0][ii,:-1]==interpolate_emb[0][ii+1,:-1]).all():
                        save_pos.append(ii)

            savepos2=torch.Tensor(save_pos)
            save_edit.append([interp.clone(),interp_len.clone(),savepos2.clone()])



            for j in range(len(save_edit)):

                path_name='../Edit_'+model_dir+'/text20_generatadvideo/{:07d}/'.format((num_train+i)*BATCH_SIZE+j)
                os.makedirs(path_name, exist_ok=True)

                np.savez(path_name+'data',people=save_edit[j][0][0].to('cpu').detach().numpy().copy(),
                gen_framelen=(int)(save_edit[j][1].to('cpu').detach().numpy().copy()),
                savepos=save_edit[j][2].to('cpu').detach().numpy().copy())



            save_edit=[]
            savepos_list=[]
            save_edit2=[]
            savepos_list2=[]
            interpolate_emb_p=inputemb.unsqueeze(-1).repeat(1,1,2,1).view(inputemb.size(0),inputemb.size(1)*2,inputemb.size(2))
            interp_len_p=framelen*2-1
            frame=model.dataM.emb_to_data(inputemb,interp_len_p,None,False)
            interp=model.dataM.emb_to_data(interpolate_emb_p,interp_len_p,None,False)
            for i in range(len(inputemb)):
                interp_len=interp_len_p[i]
                savepos=[]
                for ii in range((int)(framelen[i])-1):
                    if not (inputemb[i,ii,:-1]==inputemb[i,ii+1,:-1]).all():
                        savepos.append(ii)
                savepos2=torch.Tensor(savepos)

                save_edit.append([frame[i:i+1].clone(),framelen[i:i+1].clone(),savepos2.clone()])
                savepos=[]
                for ii in range((int)(interp_len)-1):
                    if not (interpolate_emb_p[i][ii,:-1]==interpolate_emb_p[i][ii+1,:-1]).all():
                        savepos.append(ii)

                savepos2=torch.Tensor(savepos)
                save_edit2.append([interp[i:i+1].clone(),interp_len.clone(),savepos2.clone()])
            for j in range(len(save_edit)):

                path_name='../Orig_'+model_dir+'/text20_generatadvideo/orig{:07d}/'.format((num_train+i)*BATCH_SIZE+j)
                os.makedirs(path_name, exist_ok=True)

                np.savez(path_name+'data',people=save_edit[j][0][0].to('cpu').detach().numpy().copy(),
                gen_framelen=(int)(save_edit[j][1].to('cpu').detach().numpy().copy()),savepos=save_edit[j][2].to('cpu').detach().numpy().copy())

                path_name='../Edit_'+model_dir+'/text20_generatadvideo/interp{:07d}/'.format((num_train+i)*BATCH_SIZE+j)
                os.makedirs(path_name, exist_ok=True)

                np.savez(path_name+'data',people=save_edit2[j][0][0].to('cpu').detach().numpy().copy(),
                gen_framelen=(int)(save_edit2[j][1].to('cpu').detach().numpy().copy()),savepos=save_edit2[j][2].to('cpu').detach().numpy().copy())

            break
