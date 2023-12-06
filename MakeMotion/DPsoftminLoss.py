

import torch
BISHO=1.0e-10
global path
global match_pos2
P_PENAL=3.2
temperature2=0.01
P_DIFFRAME3=1
P_DIFFRAME4=1
WrongWeightPenal=3.2
softOneHotRandom=1.0
Reconstruct_weight=0.0
gamma=0.01
import torch.nn as nn

class DPsoftminLoss(nn.Module):
    def __init__(self,D_model_d):
        super(DPsoftminLoss, self).__init__()
        self.dim=D_model_d

    def forward(self,DPmap,frame,framelen):
        frame=frame.unsqueeze(2)

        DPLoss=torch.zeros(DPmap.size(0),DPmap.size(1),DPmap.size(2)).to(DPmap.device)

        DPLoss=DPLoss+torch.sum(torch.abs(frame[:,:,:,:self.dim]-DPmap[:,:,:,:self.dim]),-1)
        costMap=torch.zeros(DPLoss.size()).to(DPmap.device)
        costMap2=torch.zeros(DPLoss.size()).to(DPmap.device)
        costMap[:,0,0]=DPLoss[:,0,0]
        costMap2[:,0,0]=DPLoss[:,0,0]
        for i in range(1,DPmap.size(1)):
            #ln=nn.LayerNorm(i, eps=1e-6)
            #tmp=torch.exp(-1*(ln((costMap[:,i-1,:i]-torch.mean(costMap[:,i-1,:i],1).unsqueeze(1)).to("cpu")))/temperature2).detach().to(DPmap.device)
            #print(costMap[:,i-1])
            #print(tmp)
            #print(tmp[0]

            costMap[:,i,:i]=costMap[:,i-1,:i].clone()+DPLoss[:,i,:i]

            #costMap_temp=softOneHotRandom*torch.sum(tmp[:,:i]*costMap[:,i-1,:i].clone(),1)/torch.sum(tmp[:,:i],1) + (1-softOneHotRandom)/i*torch.sum(costMap[:,i-1,:i].clone(),1)
            #costMap[:,i,i]=torch.min(costMap[:,i-1,:i],1)[0]+(costMap_temp-costMap_temp.detach())+P_PENAL+DPLoss[:,i,i]
            costMap_temp=temperature2*-1*torch.log(torch.sum(torch.exp(-1*(costMap[:,i-1,:i]-torch.min(costMap[:,i-1,:i],-1)[0].unsqueeze(-1)).clone()/temperature2),-1))+P_PENAL+DPLoss[:,i,i]+torch.min(costMap[:,i-1,:i],-1)[0]# + (1-softOneHotRandom)/i*torch.sum(costMap[:,i-1,:i].clone(),1)
            #costMap_temp=temperature2*-1*torch.log(torch.sum(torch.exp(-1*costMap[:,i-1,:i].clone()/temperature2),-1))+P_PENAL+DPLoss[:,i,i]# + (1-softOneHotRandom)/i*torch.sum(costMap[:,i-1,:i].clone(),1)
            costMap[:,i,i]=costMap_temp
            costMap2[:,i,:i]=costMap2[:,i-1,:i].clone()+DPLoss[:,i,:i]
            costMap2[:,i,i]=torch.min(costMap2[:,i-1,:i],1)[0]+P_PENAL+DPLoss[:,i,i]

            '''
            for j in range(DPmap.size(0)):

                if torch.isnan(costMap[j,i,i]).any():
                    if torch.isnan(torch.sum(tmp[:,:i]*costMap[:,i-1,:i].clone(),1)).any() or torch.isinf(torch.sum(tmp[:,:i]*costMap[:,i-1,:i].clone(),1)).any():
                        print(tmp[j,:i])
                        print(costMap[j,i-1,:i])
                        print(ln(costMap[:,i-1,:i])[j])
                        ln2=nn.LayerNorm(i, eps=1e-6)
                        print(ln2(costMap[:,i-1,:i]).to("cpu"))
                        print("1")
                    elif torch.isnan(torch.sum(tmp[:,:i]*costMap[:,i-1,:i].clone(),1)/torch.sum(tmp[:,:i],1)).any() or torch.isinf(torch.sum(tmp[:,:i]*costMap[:,i-1,:i].clone(),1)/torch.sum(tmp[:,:i],1)).any():
                        print(tmp[j,:i])
                        print(costMap[j,i-1,:i])
                        print("3")

                    if torch.isnan(DPLoss[j,i,i]).any() or torch.isinf(DPLoss[j,i,i]).any():
                        print(DPLoss[j,i,i])
                        print("2")
                    print("cost",i,j)
                    exit()
                    '''


        k=torch.gather(costMap,1,(framelen-1).unsqueeze(1).unsqueeze(2).repeat(1,1,costMap.size(2))).squeeze(1)
        trilTensor=torch.tril(torch.ones(costMap.size(1),costMap.size(1))).to(costMap.device)
        '''
        tmptensor=torch.exp(-1*(k-torch.max(k,1)[0].unsqueeze(1))/temperature2).detach()

        tmpsum=torch.sum(trilTensor[framelen-1]*tmptensor,1)
        ans=torch.sum(k*tmptensor/tmpsum.unsqueeze(1))
        '''
        ans=0
        for i in range(DPmap.size(0)):
            if framelen[i]==1:
                tmptensor=torch.ones(framelen[i]).to(DPmap.device)
            else:
                ln2=nn.LayerNorm((int)(framelen[i]), eps=1e-6)
                tmptensor= torch.exp(-1*(ln2((k[i,:framelen[i]]-torch.mean(k[i,:framelen[i]])).to("cpu")))/temperature2).detach().to(DPmap.device)

            #ans+=torch.sum(k[i,:framelen[i]]*(tmptensor*softOneHotRandom/torch.sum(tmptensor) + (1-softOneHotRandom)/framelen[i]))
            ans+=temperature2*-1*torch.log(torch.sum(torch.exp(-1*(k[i,:framelen[i]]-torch.min(k[i,:framelen[i]],-1)[0].unsqueeze(-1)).clone()/temperature2)))+torch.min(k[i,:framelen[i]],-1)[0]
            #ans+=temperature2*-1*torch.log(torch.sum(torch.exp(-1*k[i,:framelen[i]].clone()/temperature2)))
            tmp_reconstruct=torch.diagonal(DPLoss,dim1=1,dim2=2)
            ans+=Reconstruct_weight*torch.sum(tmp_reconstruct-tmp_reconstruct.detach())

        k2=torch.gather(costMap2,1,(framelen-1).unsqueeze(1).unsqueeze(2).repeat(1,1,costMap.size(2))).squeeze(1)
        lastindexEmb=torch.gather(torch.cummin(k2,1)[1],1,(framelen-1).unsqueeze(1)).squeeze(1)#index
        savepos=torch.eye(costMap.size(1))[framelen-1].unsqueeze(2)*torch.eye(costMap.size(2))[lastindexEmb].unsqueeze(1)
        savepos=savepos.to(DPmap.device)
        eyeTensor=torch.eye(costMap.size(1)).to(DPmap.device)
        for i in range(DPmap.size(1)-1,0,-1):
            savepos[:,i-1]=savepos[:,i-1].clone()+savepos[:,i,i:i+1].clone()*eyeTensor[torch.argmin(costMap2[:,i-1,:i],-1)]+savepos[:,i].clone()
            switchpos=torch.where( torch.diagonal(savepos,dim1=2,dim2=1)==torch.Tensor([1]).to(savepos.device), torch.LongTensor([1]).to(savepos.device), torch.LongTensor([0]).to(savepos.device))  #0 if not, and 1 if switched

        #print(switchpos[0,:framelen[0]])
        return ans,switchpos
