import sys
from os.path import dirname, realpath
import torch
import torch.utils.data as data
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.ubuntu_data_utils as ubuntu_data_utils

class AskUbuntuDataset(data.Dataset):
    def __init__(self, name, word_to_indx, max_seq_length=100, training_data_size=200):
        self.name = name
        self.dataset = []
        self.word_to_indx = word_to_indx
        self.max_seq_length = max_seq_length
        self.data_dict = ubuntu_data_utils.get_data_dict()

        if name == 'train':
            train_examples = ubuntu_data_utils.get_train_examples()[:training_data_size]
            for example in train_examples:
                self.update_dataset_from_train_example(example)
        elif name == 'dev':
            dev_examples = ubuntu_data_utils.get_dev_examples()
            for example in dev_examples:
                self.update_dataset_from_dev_or_test_example(example)
        elif name == 'test':
            test_examples = ubuntu_data_utils.get_test_examples()
            for example in test_examples:
                self.update_dataset_from_dev_or_test_example(example)
        else:
            raise Exception("Data set name {} not supported!".format(name))

    ## Convert one example to {x: example, y: label (always 0)}
    def update_dataset_from_train_example(self, example):
        # adds samples to dataset for each training example
        # each training example generates multiple samples
        qid, similar_qids, random_qids = example
        qid_tensors = map(self.get_indices_tensor, self.data_dict[qid])

        random_candidate_tensors = [map(self.get_indices_tensor, self.data_dict[cqid]) for cqid in random_qids]
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

    def update_dataset_from_dev_or_test_example(self, example):
        qid, similar_qids, candidate_qids, BM25_scores = example
        qid_tensors = map(self.get_indices_tensor, self.data_dict[qid])

        # candidate_tensors = [map(self.get_indices_tensor, self.data_dict[cqid]) for cqid in candidate_qids]
        # candidate_title_tensors, candidate_body_tensors = zip(*candidate_tensors)
        # candidate_title_tensors = list(candidate_title_tensors)
        # candidate_body_tensors = list(candidate_body_tensors)

        labels = [1 if cqid in similar_qids else 0 for cqid in candidate_qids]

        positive_qids = similar_qids
        positive_tensors = [map(self.get_indices_tensor, self.data_dict[qid]) for qid in positive_qids]
        if len(positive_tensors) > 0:
            positive_title_tensors, positive_body_tensors = zip(*positive_tensors)
            positive_title_tensors = list(positive_title_tensors)
            positive_body_tensors = list(positive_body_tensors)
        else:
            positive_title_tensors, positive_body_tensors = [], []

        negative_qids = [cpid for cpid in candidate_qids if cpid not in similar_qids]
        negative_tensors = [map(self.get_indices_tensor, self.data_dict[qid]) for qid in negative_qids]
        if len(negative_tensors) > 0:
            negative_title_tensors, negative_body_tensors = zip(*negative_tensors)
            negative_title_tensors= list(negative_title_tensors)
            negative_body_tensors= list(negative_body_tensors)
        else:
            negative_title_tensors, negative_body_tensors = [], []

        sample = \
            {'qid': qid,
             'similar_qids': similar_qids,
             'candidates': candidate_qids,
             'q_title_tensor': qid_tensors[0],
             'q_body_tensor': qid_tensors[1],
             # 'candidate_title_tensors': candidate_title_tensors,
             # 'candidate_body_tensors': candidate_body_tensors,
             'positive_title_tensors': positive_title_tensors,
             'positive_body_tensors': positive_body_tensors,
             'negative_title_tensors': negative_title_tensors,
             'negative_body_tensors': negative_body_tensors,
             'BM25_scores': BM25_scores,
             'labels': labels
             }
        self.dataset.append(sample)
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample

    def get_indices_tensor(self, text_arr):
        nil_indx = 0
        unk_indx = 1
        text_indx = [self.word_to_indx[x.lower()] if x.lower() in self.word_to_indx else unk_indx for x in
                     text_arr.split()][:self.max_seq_length]

        if len(text_indx) < self.max_seq_length:
            text_indx.extend([nil_indx for _ in range(self.max_seq_length - len(text_indx))])
        x = torch.LongTensor(text_indx)
        return x