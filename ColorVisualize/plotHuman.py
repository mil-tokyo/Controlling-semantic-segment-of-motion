import sys
shownum=4648
data_folder=sys.argv[1]
use_mean=sys.argv[2]
import os
import glob


import utils.paramUtil as paramUtil

from utils.plot_script import *


from scripts.motion_process import *




def plot_t2m(data, save_dir, captions,std,mean,savepos):
    if use_mean=="1":
        data = data * std + mean
    if True:
        i=0
        caption=" "
        joint_data=data[0]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(),  22).numpy()
        save_path = '%s_%02d'%(save_dir, i)
        plot_3d_motion(save_path + '.mp4', paramUtil.t2m_kinematic_chain, joint, savepos,caption, fps=20)
        print(caption)

os.makedirs("out_"+data_folder+"/", exist_ok=True)
if use_mean=="1":
    mean = np.load('../'+data_folder+'/checkpoints/t2m/Comp_v6_KLD01/meta/mean.npy')#np.load('../../../dataset/HumanML3D/HumanML3D/Mean.npy')
    std = np.load('../'+data_folder+'/checkpoints/t2m/Comp_v6_KLD01/meta/std.npy')
else:
    mean=None
    std=None

if True:
    file_list = sorted(glob.glob("../"+data_folder+"/"+'/text20_generatadvideo/*/'))
    if use_mean=="1":
        print(file_list[0:4])

        index_list=np.load('../index_list.npz')['index_list']

    for ii in range(shownum):
        i=ii
        if use_mean=="1":
            i=index_list[ii]
        os.makedirs("out_"+data_folder+'/{:07d}/'.format(ii), exist_ok=True)
        captions=None
        data=np.load(file_list[i]+"data.npz")
        print(file_list[i])
        if 'people' in data:
            plot_t2m(data["people"][np.newaxis,:data['gen_framelen'],:], pjoin("out_"+data_folder+'/{:07d}/'.format(ii), 'gen_motion_L%03d' % ((int)(data["gen_framelen"]))), captions,std,mean,data["savepos"])
