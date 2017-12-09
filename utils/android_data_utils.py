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
GLOVE_FILE = os.path.join(HOME_PATH,'data/pruned_glove.840B.300d.txt')


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

def get_test_examples():
    return get_examples(TEST_POS_FILE, TEST_NEG_FILE)

def get_dev_examples():
    return get_examples(DEV_POS_FILE, DEV_NEG_FILE)

