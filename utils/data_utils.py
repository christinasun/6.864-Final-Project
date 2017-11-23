import os
import numpy as np
from os.path import dirname, realpath
import gzip
import torch
import torch.utils.data as data


HOME_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(HOME_PATH,'data')
VECTORS_FILE = os.path.join(DATA_PATH,"vector","vectors_pruned.200.txt.gz")
DATA_FILE = os.path.join(DATA_PATH,"text_tokenized.txt")
TRAIN_FILE = os.path.join(DATA_PATH,"train_random.txt")
DEV_SET_FILE = os.path.join(DATA_PATH,"dev.txt")
TEST_SET_FILE = os.path.join(DATA_PATH,"test.txt")

# TODO figure out when the people in the paper threw away examples
class AskUbuntuDataset(data.Dataset):

    # TODO: modify the max_length based on the specifications in the paper
    def __init__(self, name, word_to_indx, max_length=200, max_dataset_size=500):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.data_dict = get_data_dict()

        if name == 'train':
            train_examples = get_train_examples()[:max_dataset_size]
            for example in train_examples:
                self.update_dataset_from_train_example(example)
        #TODO: implement 'dev' and 'test'
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

    def get_indices_tensor(self, text_arr):
        nil_indx = 0
        text_indx = [ self.word_to_indx[x] if x in self.word_to_indx else nil_indx for x in text_arr][:self.max_length]
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
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1

    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

def get_data_dict():
    data_dict = {}

    with open(DATA_FILE,"rb") as f:
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
