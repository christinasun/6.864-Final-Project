import os
import numpy as np
from os.path import dirname, realpath
import gzip
import torch
import torch.utils.data as data


HOME_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(HOME_PATH,'data','ubuntu')
VECTORS_FILE = os.path.join(DATA_PATH,"vector","vectors_pruned.200.txt.gz")
DATA_FILE = os.path.join(DATA_PATH,"text_tokenized.txt.gz")
TRAIN_FILE = os.path.join(DATA_PATH,"train_random.txt")
DEV_SET_FILE = os.path.join(DATA_PATH,"dev.txt")
TEST_SET_FILE = os.path.join(DATA_PATH,"test.txt")

# TODO figure out when the people in the paper threw away examples
# TODO determine what we want the max length to be and if we want to make it configurable using an argument
# TODO determine if we want to add the option to use titles only
class AskUbuntuDataset(data.Dataset):

    # TODO: modify the max_length based on the specifications in the paper
    def __init__(self, name, word_to_indx, max_length=100, trainning_data_size=200):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.data_dict = get_data_dict()

        if name == 'train':
            train_examples = get_train_examples()[:trainning_data_size]
            for example in train_examples:
                self.update_dataset_from_train_example(example)
        elif name == 'dev':
            dev_examples = get_dev_examples()
            for example in dev_examples:
                self.update_dataset_from_dev_or_test_example(example)
        elif name == 'test':
            test_examples = get_test_examples()
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

    def update_dataset_from_dev_or_test_example(self, example):
        # adds samples to dataset for each training example
        # each training example generates multiple samples
        qid, similar_qids, candidate_qids, BM25_scores  = example
        qid_tensors = map(self.get_indices_tensor, self.data_dict[qid])

        candidate_tensors = [map(self.get_indices_tensor,self.data_dict[cqid]) for cqid in candidate_qids]
        candidate_title_tensors, candidate_body_tensors = zip(*candidate_tensors)
        candidate_title_tensors = list(candidate_title_tensors)
        candidate_body_tensors = list(candidate_body_tensors)

        labels = [1 if cqid in similar_qids else 0 for cqid in candidate_qids]

        sample = \
            {'qid': qid,
             'similar_qids': similar_qids,
             'candidates': candidate_qids,
             'qid_title_tensor': qid_tensors[0],
             'qid_body_tensor': qid_tensors[1],
             'candidate_title_tensors': candidate_title_tensors,
             'candidate_body_tensors': candidate_body_tensors,
             'BM25_scores': BM25_scores,
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
        if len(text_indx) < self.max_length:
            text_indx.extend( [nil_indx for _ in range(self.max_length - len(text_indx))])
        x =  torch.LongTensor(text_indx)
        return x


def get_embeddings_tensor():
    with gzip.open(VECTORS_FILE) as f:
        content = f.readlines()

    embedding_tensor = []
    word_to_indx = {}
    for indx, line in enumerate(content):
        word, vector_string = line.strip().split(" ", 1)
        vector = map(float, vector_string.split(" "))

        if indx == 0:
            embedding_tensor.append(np.zeros(len(vector)))
            embedding_tensor.append(np.zeros(len(vector)))
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+2

    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

def get_data_dict():
    data_dict = {}

    with gzip.open(DATA_FILE) as f:
        content = f.readlines()
        for line in content:
            qid, question = line.strip().split("\t",1)
            question = question.split("\t")
            if len(question) == 1: #there's only a title
                data_dict[int(qid)] = (question[0], "")
            elif len(question) == 2: #there's a title and a body
                data_dict[int(qid)] = (question[0], question[1])
            else: # There's some problem with the dataset
                assert False
    return data_dict

def get_train_examples():
    # list of tuples of the form (qid, similar qids, randomly selected qids)
    train_examples = []

    with open(TRAIN_FILE,"rb") as f:
        content = f.readlines()
        for line in content:
            qid, similar_qids, random_qids = line.strip().split("\t",2)
            similar_qids = map(int, similar_qids.split(' '))
            random_qids = map(int, random_qids.split(' '))
            train_examples.append((int(qid), similar_qids, random_qids))

    return train_examples

def get_examples_helper(dataset_file):
    # list of tuples of the form (qid, similar qids, candidates,  associated BM25 scores)
    examples = []

    with open(dataset_file,"rb") as f:
        content = f.readlines()
        for line in content:
            qid, similar_qids, candidates, bm25_scores = line.strip().split("\t",3)
            similar_qids = [] if similar_qids == "" else map(int, similar_qids.split(' '))
            candidates = map(int, candidates.split(' '))
            bm25_scores = map(float, bm25_scores.split(' '))
            examples.append((int(qid), similar_qids, candidates, bm25_scores))
    return examples


def get_test_examples():
    return get_examples_helper(TEST_SET_FILE)


def get_dev_examples():
    return get_examples_helper(DEV_SET_FILE)
