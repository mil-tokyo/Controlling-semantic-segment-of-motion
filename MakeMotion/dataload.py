import glob
import torch
import numpy as np


import numpy as np
import pickle
from os.path import join as pjoin


import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
#import spacy

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}



from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
bonelist=['mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head','mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm','mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm','mixamorig:LeftUpLeg', 'mixamorig:LeftLeg', 'mixamorig:RightUpLeg', 'mixamorig:RightLeg']

class WordVectorizer(object):
    def __init__(self):
        vectors = np.load('glove/our_vab_data.npy')
        words = pickle.load(open('glove/our_vab_words.pkl', 'rb'))
        word2idx = pickle.load(open('glove/our_vab_idx.pkl', 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec


w_vectorizer=WordVectorizer()

class Text2MotionDataset_test(torch.utils.data.Dataset):
    def __init__(self, split_file):
        #self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40# if self.opt.dataset_name =='t2m' else 24

        #joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        self.motion=[]
        self.labellen=[]
        self.datalen=[]
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin('../../../dataset/HumanML3D/HumanML3D/new_joint_vecs', name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                motion=np.insert(motion,motion.shape[1],0,axis=1)
                motion[-1][-1]=1
                text_data = []
                flag = False
                with cs.open(pjoin('../../../dataset/HumanML3D/HumanML3D/texts', name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        with open("caption_index.txt") as f:
            index_list_caption = f.read().split(" ")
        cnt=0
        for i in new_name_list:
            cnt2 = 0
            for j in data_dict[i]['text']:
                if cnt2 == index_list_caption[cnt]:
                    data_dict[i]['text']=[j]
                cnt2 += 1



            cnt+=1




        #name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list=new_name_list



        self.data_dict = data_dict
        self.name_list = name_list
        self.textstore=[]

        for i in name_list:
            data = self.data_dict[i]
            motion, m_length, text_list = data['motion'], data['length'], data['text']
            text_data = random.choice(text_list)
            caption, tokens = text_data['caption'], text_data['tokens']
            self.datalen.append(len(data['motion']))
            self.labellen.append(len(tokens)+2)
            self.motion.append(torch.from_numpy(data['motion']))
            self.textstore.append(data['text'])
        print(name_list)

        self.motion=pad_sequence(self.motion,batch_first=True,padding_value=0)
        self.maxlabellen=max(self.labellen)


    def __len__(self):
        return len(self.data_dict)# - self.pointer
    def getrawtext(self,idx):
        return self.data_dict[self.name_list[idx]]["text"]['caption']

    def __getitem__(self, item):
        idx = item#self.pointer + item

        '''
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        maxlabellen=max(self.labellen)
        '''

        motion=self.motion[idx]
        m_length=self.datalen[idx]
        text_list=self.textstore[idx]
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if len(tokens) < self.maxlabellen:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.maxlabellen + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.maxlabellen]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        #len_gap = (m_length - self.max_length) // self.opt.unit_length



        #return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
        #return self.motion[idx],torch.from_numpy(word_embeddings),torch.from_numpy(pos_one_hots),torch.LongTensor([sent_len])[0],torch.LongTensor([len(motion)])[0],caption
        return motion, torch.from_numpy(word_embeddings), torch.from_numpy(pos_one_hots),  sent_len,  m_length,caption
        #return out_dataframe.float(),  out_label.float(),label_len,frame_len,time,frame_len2,path,path2



class Text2MotionDataset(torch.utils.data.Dataset):
    def __init__(self, split_file):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 #if self.opt.dataset_name =='t2m' else 24

        #joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        self.motion=[]
        self.labellen=[]
        self.datalen=[]

        for name in tqdm(id_list):

            try:
                motion = np.load(pjoin('../../../dataset/HumanML3D/HumanML3D/new_joint_vecs', name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                motion=np.insert(motion,motion.shape[1],0,axis=1)
                motion[-1][-1]=1
                #print("2",np.shape(frame))
                #frame=np.append(frame,np.append(np.ones((1,1)),np.zeros((1,2*12)),axis=1),axis=0)
                #motion[-1][4*len(bonelist)]=1
                text_data = []
                flag = False
                with cs.open(pjoin('../../../dataset/HumanML3D/HumanML3D/texts', name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))

                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break


                if flag:



                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))


            except:
                # Some motion may not exist in KIT dataset
                pass



        #name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list=new_name_list



        self.data_dict = data_dict
        self.name_list = name_list


        self.textstore=[]
        for i in name_list:
            data = self.data_dict[i]
            motion, m_length, text_list = data['motion'], data['length'], data['text']
            text_data = random.choice(text_list)
            caption, tokens = text_data['caption'], text_data['tokens']
            self.datalen.append(len(data['motion']))
            self.labellen.append(len(tokens)+2)
            self.motion.append(torch.from_numpy(data['motion']))
            self.textstore.append(data['text'])

        self.motion=pad_sequence(self.motion,batch_first=True,padding_value=0)
        self.maxlabellen=max(self.labellen)



    def __len__(self):
        return len(self.data_dict)# - self.pointer

    def getrawtext(self,idx):
        return self.data_dict[self.name_list[idx]]["text"]['caption']

    def __getitem__(self, item):
        idx = item#self.pointer + item

        '''
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        maxlabellen=max(self.labellen)
        '''

        motion=self.motion[idx]
        m_length=self.datalen[idx]
        text_list=self.textstore[idx]
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if len(tokens) < self.maxlabellen:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.maxlabellen + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.maxlabellen]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        #len_gap = (m_length - self.max_length) // self.opt.unit_length



        #return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
        #return self.motion[idx],torch.from_numpy(word_embeddings),torch.from_numpy(pos_one_hots),torch.LongTensor([sent_len])[0],torch.LongTensor([len(motion)])[0],caption
        return motion, torch.from_numpy(word_embeddings), torch.from_numpy(pos_one_hots),  sent_len,  m_length,caption
        #return out_dataframe.float(),  out_label.float(),label_len,frame_len,time,frame_len2,path,path2
