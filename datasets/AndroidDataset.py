import sys
from os.path import dirname, realpath
import torch
import torch.utils.data as data
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.android_data_utils as android_data_utils

class AndroidDataset(data.Dataset):
    def __init__(self, name, word_to_indx, max_length=100, max_dataset_size=800):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.data_dict = android_data_utils.get_data_dict()

        if name == 'dev':
            dev_examples = android_data_utils.get_dev_examples()[:max_dataset_size]
            print("dev_examples: ", len(dev_examples))
            for example in dev_examples:
                self.update_dataset_from_example(example)
        elif name == 'test':
            test_examples = android_data_utils.get_test_examples()[:max_dataset_size]
            print("test_examples: ", len(test_examples))
            for example in test_examples:
                self.update_dataset_from_example(example)
        else:
            raise Exception("Data set name {} not supported!".format(name))

    ## Convert one example to {x: example, y: label (always 0)}
    def update_dataset_from_example(self, example):
        # adds samples to dataset for each training example
        # each training example generates multiple samples
        qid, similar_qids, random_qids = example
        qid_tensors = map(self.get_indices_tensor, self.data_dict[qid])

        random_candidate_tensors = [map(self.get_indices_tensor,self.data_dict[cqid]) for cqid in random_qids]
        random_candidate_title_tensors, random_candidate_body_tensors = zip(*random_candidate_tensors)
        random_candidate_title_tensors = list(random_candidate_title_tensors)
        random_candidate_body_tensors = list(random_candidate_body_tensors)

        for similar_qid in similar_qids:
            candidates = [similar_qid] + random_qids
            similar_qid_tensors = map(self.get_indices_tensor, self.data_dict[similar_qid])

            sample = {'qid': qid,
                      'candidates': candidates,
                      'qid_title_tensor': qid_tensors[0],
                      'qid_body_tensor': qid_tensors[1],
                      'candidate_title_tensors': [similar_qid_tensors[0]] + random_candidate_title_tensors,
                      'candidate_body_tensors': [similar_qid_tensors[1]] + random_candidate_body_tensors
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

        for indx in text_indx:
            if indx in self.indx_to_count.keys():
                self.indx_to_count[indx] += 1
            else:
                self.indx_to_count[indx] = 0

        if len(text_indx) < self.max_length:
            text_indx.extend( [nil_indx for _ in range(self.max_length - len(text_indx))])
        x =  torch.LongTensor(text_indx)
        return x
