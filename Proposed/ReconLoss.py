import torch
BISHO=1.0e-10
global path
global match_pos2

bonelist=['mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head','mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm','mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm','mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg']
bonenum=len(bonelist)
import torch.nn as nn
class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()

    def forward(self,gen,gt,endframenum,framelen):

        def dist_t(p1_b,p2_b):

            p1_mask=torch.where(p1_b==float('inf'),torch.Tensor([0]).to(p1_b.device),torch.Tensor([1]).to(p1_b.device))
            p2_mask=torch.where(p2_b==float('inf'),torch.Tensor([0]).to(p1_b.device),torch.Tensor([1]).to(p1_b.device))
            mask=p1_mask*p2_mask
            p1_2b, p2_2b = torch.broadcast_tensors(p1_b, p2_b)
            p1_b2=torch.where(mask==0,mask,p1_2b)
            p2_b2=torch.where(mask==0,mask,p2_2b)
            p1=p1_b2[:,:,:-1]
            p2=p2_b2[:,:,:-1]
            p1e=p1_b2[:,:,-1:]
            p2e=p2_b2[:,:,-1:]
            difloss=torch.zeros(p1.size(0),p1.size(1)).to(p1.device)
            difloss+=torch.sum(torch.abs(p1-p2),-1)
            return difloss

        map_dist_simple=torch.cumsum(dist_t(gen,gt),1)

        ans=torch.gather(map_dist_simple.to('cpu'),1,(framelen-1).to('cpu').unsqueeze(-1)).squeeze(-1).to(map_dist_simple.device)
        return ans
