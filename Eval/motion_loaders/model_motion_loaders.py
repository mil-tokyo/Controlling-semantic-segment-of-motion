from torch.utils.data import DataLoader, Dataset
from utils.get_opt import get_opt
from motion_loaders.comp_v6_model_dataset import CompV6GeneratedDataset
from utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate
from options.base_options import seed
import torch
import random
g = torch.Generator()
g.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            if len(motion) < self.opt.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.opt.max_motion_length - len(motion), motion.shape[1]))
                                         ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens



def get_motion_loader(opt_path, dataset_load_path2,model_type,batch_size, ground_truth_dataset, mm_num_samples, mm_num_repeats, device):
    if model_type==0:
        opt = get_opt(opt_path, device)
        optname=opt.name
    elif model_type==1:
        opt = get_opt(opt_path, device)
        optname=dataset_load_path2

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    print('Generating %s ...' % optname)

    dataset = CompV6GeneratedDataset(opt, dataset_load_path2,model_type,ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)


    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,drop_last=True, num_workers=4,worker_init_fn=seed_worker,generator=g,)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1,shuffle=True,worker_init_fn=seed_worker,generator=g,)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader
