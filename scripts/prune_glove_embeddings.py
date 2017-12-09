import sys
import os
from os.path import dirname, realpath
import gzip

sys.path.append(dirname(dirname(realpath(__file__))))
import utils.android_data_utils as android_data_utils
import utils.ubuntu_data_utils as ubuntu_data_utils
import pickle as p

HOME_PATH = dirname(dirname(realpath(__file__)))
GLOVE_FILE = os.path.join(HOME_PATH,'data/glove.840B.300d.txt')
PRUNED_GLOVE_FILE = os.path.join(HOME_PATH,'data/pruned_glove.840B.300d.txt')
WORD_SET_FILE = os.path.join('data/word_set.p')
DATA_PATH = os.path.join(HOME_PATH,'data','ubuntu')
UBUNTU_EMBEDDING_FILE = os.path.join(DATA_PATH,"vector","vectors_pruned.200.txt.gz")


if __name__ == "__main__":
    ubuntu_data_dict = ubuntu_data_utils.get_data_dict()
    android_data_dict = android_data_utils.get_data_dict()

    ubuntu_word_set = set()

    for query in ubuntu_data_dict.values():
        title, body = query
        title_words = title.split()
        body_words = body.split()

        for word in title_words:
            ubuntu_word_set.add(word.lower())
        for word in body_words:
            ubuntu_word_set.add(word.lower())

    print "Length of ubuntu_word_set: {}".format(len(ubuntu_word_set))

    android_word_set = set()
    for query in android_data_dict.values():
        title, body = query
        title_words = title.split()
        body_words = body.split()

        for word in title_words:
            android_word_set.add(word.lower())
        for word in body_words:
            android_word_set.add(word.lower())

    print "Length of android_word_set: {}".format(len(android_word_set))

    word_set = set()
    for word in android_word_set: word_set.add(word)
    for word in ubuntu_word_set: word_set.add(word)
    print "Length of word_set: {}".format(len(word_set))


    # with open(WORD_SET_FILE,'wb+') as f:
    #     p.dump(word_set,f)

    counter = 0
    saved_counter = 0
    with open(GLOVE_FILE) as glove_file:
        with open(PRUNED_GLOVE_FILE,'w+') as pruned_glove_file:
            for line in glove_file:
                counter += 1
                word, vector_string = line.strip().split(" ", 1)
                if word in word_set:
                    saved_counter += 1
                    # pruned_glove_file.write(line)
                else:
                    pass
                    # print word

    print "glove counter: {}".format(counter)
    print "saved glove counter: {}".format(saved_counter)


    counter = 0
    saved_counter = 0
    with gzip.open(UBUNTU_EMBEDDING_FILE) as ubuntu_embedding_file:
        for line in ubuntu_embedding_file:
            counter += 1
            word, vector_string = line.strip().split(" ", 1)
            if word in word_set:
                saved_counter += 1
            else:
                pass
                # print word

    print "ubuntu counter: {}".format(counter)
    print "saved ubuntu counter: {}".format(saved_counter)








