import sys
from os.path import dirname, realpath
import torch
import torch.utils.data as data
import utils.android_data_utils as android_data_utils
import utils.ubuntu_data_utils as ubuntu_data_utils
import random
import numpy as np
sys.path.append(dirname(dirname(realpath(__file__))))

class TransferDataset(data.Dataset):
    def __init__(self, name, word_to_indx, max_seq_length=100, max_dataset_size=800):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_seq_length
        self.dataset_size = max_dataset_size
        self.num_samples = 20
        self.android_data_dict = android_data_utils.get_data_dict()
        self.ubuntu_data_dict = ubuntu_data_utils.get_data_dict()

        for i in xrange(self.dataset_size):
            self.update_dataset_from_train_example(self.num_samples)

    def update_dataset_from_train_example(self, num_samples):

        android_ids = random.sample(self.android_data_dict.keys(), num_samples)
        android_tensors = [map(self.get_indices_tensor,self.android_data_dict[android_id]) for android_id in android_ids]
        android_title_tensors, android_body_tensors = zip(*android_tensors)
        android_title_tensors = list(android_title_tensors)
        android_body_tensors = list(android_body_tensors)

        ubuntu_ids = random.sample(self.ubuntu_data_dict.keys(), num_samples)
        ubuntu_tensors = [map(self.get_indices_tensor,self.android_data_dict[android_id]) for android_id in android_ids]
        ubuntu_title_tensors, ubuntu_body_tensors = zip(*android_tensors)
        ubuntu_title_tensors = list(ubuntu_title_tensors)
        ubuntu_body_tensors = list(ubuntu_body_tensors)

        ids = android_ids + ubuntu_ids
        title_tensors = android_title_tensors + ubuntu_title_tensors
        body_tensors = android_body_tensors + ubuntu_body_tensors
        labels = np.concatenate((np.zeros((num_samples,), dtype=np.int),np.ones((num_samples,), dtype=np.int)),0) 
        indexes = range(2*num_samples)
        random.shuffle(indexes)

        sample = {'id': [id for _, id in sorted(zip(indexes, ids))],
                  'title_tensors': [title_tensor for _,title_tensor in sorted(zip(indexes, title_tensors))],
                  'body_tensors': [body_tensor for _,body_tensor in sorted(zip(indexes, body_tensors))],
                  'labels': [label for _,label in sorted(zip(indexes, labels))]
                  }
        self.dataset.append(sample)
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

    def get_indices_tensor(self, text_arr):
        nil_indx = 0
        unk_indx = 1
        text_indx = [self.word_to_indx[x.lower()] if x.lower() in self.word_to_indx else unk_indx for x in text_arr.split()][:self.max_length]

        if len(text_indx) < self.max_length:
            text_indx.extend( [nil_indx for _ in range(self.max_length - len(text_indx))])
        x =  torch.LongTensor(text_indx)
        return x