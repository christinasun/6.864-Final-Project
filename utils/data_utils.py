import os
from os.path import dirname, realpath

HOME_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(HOME_PATH,'data')
VECTORS_FILE = os.path.join(DATA_PATH,"vector","vectors_pruned.200.txt")
DATA_FILE = os.path.join(DATA_PATH,"text_tokenized.txt")

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


def get_embedding_for_word(word,embeddings_dict):
    return embeddings_dict(word)
