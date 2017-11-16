import os
from os.path import dirname, realpath

HOME_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(HOME_PATH,'data')
VECTORS_FILE = os.path.join(DATA_PATH,"vector","vectors_pruned.200.txt")
DATA_FILE = os.path.join(DATA_PATH,"text_tokenized.txt")
TRAIN_FILE = os.path.join(DATA_PATH,"train_random.txt")
DEV_SET_FILE = os.path.join(DATA_PATH,"dev.txt")
TEST_SET_FILE = os.path.join(DATA_PATH,"test.txt")

def get_embeddings_dict():
    embeddings_dict = {}

    with open(VECTORS_FILE,"rb") as f:
        content = f.readlines()
        for line in content:
            word, vector_string = line.strip().split(" ",1)
            vector = map(float, vector_string.split(" "))
            embeddings_dict[word] = vector

    return embeddings_dict

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

def get_embedding_for_word(word,embeddings_dict):
    return embeddings_dict(word)
