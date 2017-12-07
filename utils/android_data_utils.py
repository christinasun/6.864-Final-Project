import os
import numpy as np
from os.path import dirname, realpath
import gzip
import torch
import torch.utils.data as data
import ubuntu_data_utils


HOME_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(HOME_PATH,'data/android')
DATA_FILE = os.path.join(DATA_PATH,"corpus.tsv")
DEV_POS_FILE = os.path.join(DATA_PATH,"dev.pos.txt")
DEV_NEG_FILE = os.path.join(DATA_PATH,"dev.neg.txt")
TEST_POS_FILE = os.path.join(DATA_PATH,"test.pos.txt")
TEST_NEG_FILE = os.path.join(DATA_PATH,"test.neg.txt")
GLOVE_FILE = os.path.join(HOME_PATH,'data/glove.6B.200d.txt')


class AndroidDataset(data.Dataset):

    def __init__(self, name, word_to_indx, max_length=100, max_dataset_size=600):
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.data_dict = get_data_dict()
        self.indx_to_count = {}

        if name == 'dev':
            dev_examples = get_examples(DEV_POS_FILE, DEV_NEG_FILE)[:max_dataset_size]
            print("dev_examples: ", len(dev_examples))
            for example in dev_examples:
                self.update_dataset_from_example(example)
        elif name == 'test':
            test_examples = get_examples(TEST_POS_FILE, TEST_NEG_FILE)[:max_dataset_size]
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

def get_examples(pos_file, neg_file):
    # list of tuples of the form (qid, positive samples, negative samples)
    samples = []
    pos_samples = get_samples(pos_file)
    neg_samples = get_samples(neg_file)

    for k in neg_samples.keys():
        samples.append((k, pos_samples[k], neg_samples[k]))
    return samples

def get_samples(file_name):
    samples = {}
    with open(file_name,"rb") as f:
        content = f.readlines()
        for line in content:
            qid, sample = line.strip().split(" ")
            if int(qid) not in samples.keys():
                samples[int(qid)] = [int(sample)]
            else:
                samples[int(qid)].append(int(sample))
    return samples

def get_embeddings_tensor():
    with open(GLOVE_FILE) as f:
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

embeddings, word_to_indx = get_embeddings_tensor()
android_dev_data = AndroidDataset('dev', word_to_indx, max_length=100)
android_test_data = AndroidDataset('test', word_to_indx, max_length=100)
print("indx_to_count_dev: ", len(android_dev_data.indx_to_count.keys()))
print "len android_dev_data {}".format(len(android_dev_data))
print "len android_test_data {}".format(len(android_test_data))

