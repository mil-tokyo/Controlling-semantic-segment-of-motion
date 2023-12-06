import torch
from networks.modules import *
from networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm

import sys
def build_models(opt,dataset_folder):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    checkpoints = torch.load(pjoin(dataset_folder, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()
    print("c48",dataset_folder,opt.dataset_name)

    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator


class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt,  dataset_folder,model_type,dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        self.model_type=model_type
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6

        if model_type==0:
            dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

            text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt,dataset_folder)
            trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
            epoch, it, sub_ep, schedule_len = trainer.load(pjoin(dataset_folder+'/t2m/Comp_v6_KLD01/model', opt.which_epoch + '.tar'))
            trainer.eval_mode()
            trainer.to(opt.device)
            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloader)):
                    word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data

                    tokens = tokens[0].split('_')
                    word_emb = word_emb.detach().to(opt.device).float()
                    pos_ohot = pos_ohot.detach().to(opt.device).float()

                    pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                    pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                    mm_num_now = len(mm_generated_motions)
                    is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                    repeat_times = mm_num_repeats if is_mm else 1
                    mm_motions = []
                    mean = np.load(dataset_folder+"/"+opt.dataset_name+'/Comp_v6_KLD01/meta/mean.npy')#np.load('../../../dataset/HumanML3D/HumanML3D/Mean.npy')
                    std = np.load(dataset_folder+"/"+opt.dataset_name+'/Comp_v6_KLD01/meta/std.npy')

                    for t in range(repeat_times):
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                        if mov_length < min_mov_length:
                            mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                        if mov_length < min_mov_length:
                            mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                        m_lens = mov_length * opt.unit_length
                        pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                              m_lens[0]//opt.unit_length, opt.dim_pose)

                        pred_motions=  pred_motions.cpu().numpy()[0] * std + mean
                        if t == 0:
                            sub_dict = {'motion': pred_motions,#.cpu().numpy(),
                                        'length': m_lens[0].item(),
                                        'cap_len': cap_lens[0].item(),
                                        'caption': caption[0],
                                        'tokens': tokens}
                            generated_motion.append(sub_dict)

                        if is_mm:
                            mm_motions.append({
                                'motion': pred_motions,#.cpu().numpy(),
                                'length': m_lens[0].item()
                            })
                    if is_mm:
                        mm_generated_motions.append({'caption': caption[0],
                                                     'tokens': tokens,
                                                     'cap_len': cap_lens[0].item(),
                                                     'mm_motions': mm_motions})

            self.generated_motion = generated_motion
            self.mm_generated_motion = mm_generated_motions
            self.opt = opt
            self.w_vectorizer = w_vectorizer
        elif model_type==1:
            BS=100
            device=opt.device
            dataloader = DataLoader(dataset, batch_size=BS, num_workers=1, shuffle=False)

            sys.path.append("../"+dataset_folder)

            from Models import ModelNetwork,D_model_f,videomaxlen,END_THRE
            import dataload
            model=ModelNetwork()
            D_model_f=D_model_f
            videomaxlen=videomaxlen
            model.load_state_dict(torch.load("../"+dataset_folder+"/text20_res_model.pt",map_location=torch.device('cpu')))
            model.eval()
            model.to(device)



            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloader)):
                    word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                    word_emb2, pos_ohot2,motions2=(word_emb).float(),(pos_ohot).float(),(motions).float()
                    cap_lens2, m_lens2=torch.LongTensor(cap_lens),torch.LongTensor(m_lens)
                    word_emb2,pos_ohot2,motions2,cap_lens2,m_lens=word_emb2.to(device),pos_ohot2.to(device),motions2.to(device),cap_lens2.to(device),m_lens.to(device)
                    word_emb = word_emb.detach().to(opt.device).float()
                    pos_ohot = pos_ohot.detach().to(opt.device).float()


                    mm_num_now = len(mm_generated_motions)
                    mm_motions = []

                    overlap_list=[]
                    for j in range(i*BS,i*BS+motions.size(0)):
                        if j in mm_idxs:
                            overlap_list.append(j)
                    if len(overlap_list)==0:
                        repeat_times=1
                    else:
                        repeat_times = mm_num_repeats


                    for t in range(repeat_times):
                        inputemb=torch.zeros(motions2.size(0),1,D_model_f).to(device)
                        input=torch.zeros(motions2.size(0),videomaxlen+1,D_model_f).to(device)
                        framelen=torch.zeros(1).to(device)
                        inputemb2, endframelist,klloss=model.textM.attn(inputemb,word_emb2,pos_ohot2,input,cap_lens2,m_lens2,0)
                        endframelist=[]
                        for en in range(motions2.size(0)):
                            tl=torch.where(inputemb2[en,:,-1]>END_THRE)[0]
                            if(len(tl)==0):
                                endframenum=(len(inputemb2[0]))
                            else:
                                endframenum=(int(tl[0])+1)
                            endframenum=max(endframenum,opt.unit_length)
                            endframelist.append(endframenum)
                        pred_motions=model.dataM.emb_to_data(inputemb2,endframelist,None,False)[:,:,:-1]
                        #pred_motions=pred_motions[:,:endframenum]
                        if t == 0:
                            # print(m_lens)
                            # print(text_data)


                            for j in range(pred_motions.size(0)):
                                token = tokens[j].split('_')
                                sub_dict = {'motion': pred_motions[j].cpu().numpy(),
                                            'length': endframelist[j],
                                            'cap_len': cap_lens[j].item(),
                                            'caption': caption[j],
                                            'tokens': token}
                                generated_motion.append(sub_dict)

                            mm_motions=[[] for j in range(len(overlap_list))]


                        for j in range(len(overlap_list)):
                            mm_motions[j].append({
                                'motion': pred_motions[overlap_list[j]%BS].cpu().numpy(),
                                'length': endframelist[overlap_list[j]%BS]
                            })
                    for j in range(len(overlap_list)):
                        token = tokens[overlap_list[j]%BS].split('_')
                        mm_generated_motions.append({'caption': caption[overlap_list[j]%BS],
                                                     'tokens': token,
                                                     'cap_len': cap_lens[overlap_list[j]%BS].item(),
                                                     'mm_motions': mm_motions[j]})

            self.generated_motion = generated_motion
            self.mm_generated_motion = mm_generated_motions
            self.opt = opt
            self.w_vectorizer = w_vectorizer




    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']

        sent_len = data['cap_len']
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.model_type==0:
            if m_length < self.opt.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                         ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
