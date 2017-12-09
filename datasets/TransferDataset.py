import sys
from os.path import dirname, realpath
import torch
import torch.utils.data as data
import utils.android_data_utils as android_data_utils
import utils.ubuntu_data_utils as ubuntu_data_utils
import random
sys.path.append(dirname(dirname(realpath(__file__))))

class TransferDataset(data.Dataset):
    def __init__(self, name, word_to_indx, max_length=100, max_dataset_size=800):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.num_train_examples = 100
        self.android_data_dict = android_data_utils.get_data_dict()
        self.ubuntu_data_dict = ubuntu_data_utils.get_data_dict()

        if name == 'train':
            print("train_examples: ", self.num_train_examples)
            for i in range(self.num_train_examples):
                self.update_dataset_from_train_example()
        elif name == 'dev':
            dev_examples = android_data_utils.get_dev_examples()
            print("dev_examples: ", len(dev_examples))
            for example in dev_examples:
                self.update_dataset_from_dev_or_test_example(example)
        elif name == 'test':
            test_examples = android_data_utils.get_test_examples()
            print("test_examples: ", len(test_examples))
            for example in test_examples:
                self.update_dataset_from_dev_or_test_example(example)
        else:
            raise Exception("Data set name {} not supported!".format(name))


    def update_dataset_from_train_example(self):
        # adds samples to dataset for each training example
        android_candidate_title_tensors = []
        android_candidate_body_tensors = []
        ubuntu_candidate_title_tensors = []
        ubuntu_candidate_body_tensors = []
        for i in range(20):
			android_key = random.choice(self.android_data_dict.keys())
        	android_candidate_tensors = map(self.get_indices_tensor,self.android_data_dict[android_key])
        	android_candidate_title_tensor, android_candidate_body_tensor = zip(*random_candidate_tensors)

        	ubuntu_key = random.choice(self.ubuntu_data_dict.keys())
        	ubuntu_candidate_tensors = map(self.get_indices_tensor,self.ubuntu_data_dict[ubuntu_key])
        	ubuntu_candidate_title_tensor, ubuntu_candidate_body_tensor = zip(*random_candidate_tensors)

        	android_candidate_title_tensors.append((android_candidate_title_tensor, 0))
        	android_candidate_body_tensors.append((android_candidate_body_tensor, 0))
        	ubuntu_candidate_title_tensors.append((ubuntu_candidate_title_tensor, 1))
        	ubuntu_candidate_body_tensors.append((ubuntu_candidate_body_tensor, 1))

        for i in xrange(len(similar_qids)):
            sample = {'title_tensors': android_candidate_title_tensors + ubuntu_candidate_title_tensors,
                      'body_tensors': android_candidate_body_tensors + ubuntu_candidate_body_tensors
                      }
            self.dataset.append(sample)
        return


    def update_dataset_from_dev_or_test_example(self, example):
        # adds samples to dataset for each training example
        # each training example generates multiple samples
        qid, similar_qids, random_qids = example
        qid_tensors = map(self.get_indices_tensor, self.data_dict[qid])

        random_candidate_tensors = [map(self.get_indices_tensor,self.data_dict[cqid]) for cqid in random_qids]
        random_candidate_title_tensors, random_candidate_body_tensors = zip(*random_candidate_tensors)
        random_candidate_title_tensors = list(random_candidate_title_tensors)
        random_candidate_body_tensors = list(random_candidate_body_tensors)

        similar_qid_tensors = [map(self.get_indices_tensor,self.data_dict[cqid]) for cqid in similar_qids]
        similar_candidate_title_tensors, similar_candidate_body_tensors = zip(*similar_qid_tensors)
        similar_candidate_title_tensors = list(similar_candidate_title_tensors)
        similar_candidate_body_tensors = list(similar_candidate_body_tensors)

        neg_labels = [0 for cqid in random_candidate_tensors]
        labels = [1] + neg_labels

        for i in xrange(len(similar_qids)):
            sample = {'qid': qid,
                      'similar_qids': similar_qids[i],
                      'candidates': [similar_qids[i]]+random_qids,
                      'q_title_tensor': qid_tensors[0],
                      'q_body_tensor': qid_tensors[1],
                      # 'candidate_title_tensors': [similar_candidate_title_tensors[i]] + random_candidate_title_tensors,
                      # 'candidate_body_tensors': [similar_candidate_body_tensors[i]] + random_candidate_body_tensors,
                      'positive_title_tensors': similar_candidate_title_tensors,
                      'positive_body_tensors': similar_candidate_body_tensors,
                      'negative_title_tensors': random_candidate_title_tensors,
                      'negative_body_tensors': random_candidate_body_tensors,
                      'labels': labels
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
