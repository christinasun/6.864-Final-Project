import argparse
import sys
import os
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
import utils.ubuntu_data_utils as ubuntu_data_utils
import utils.baseline_train_utils as train_utils
import utils.model_utils as model_utils
import utils.evaluation_utils as evaluation_utils
from datasets import AskUbuntuDataset
from utils.misc_utils import set_seeds
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AskUbuntu Question Retrieval Network')
    # learning
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--dropout', type=float, default=0.2, help='initial learning rate [default: 0.2]')
    parser.add_argument('--margin', type=float, default=0.5, help='margin size [default: 0.5]')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train [default: 10]')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size for training [default: 20]')
    parser.add_argument('--num_negative', type=int, default=20, help='# negative examples for training [default: 20]')
    parser.add_argument('--training_data_size', type=int, default = 1000000, help='Number of training queries [default: 1000000]')
    # data loading
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
    # model
    parser.add_argument('--model_name', nargs="?", type=str, default='cnn', help="Form of model, i.e dan, rnn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=20, help='dimension of the hidden layer [default: 20]')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('--train', action='store_true', default=False, help='enable train')
    parser.add_argument('--eval', action='store_true', default=False, help='enable eval')
    # task
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--save_path', type=str, default="saved_models/default", help='Path where to dump model')
    # other
    parser.add_argument('--len_query', type=int, default=100, help='how much of the query body to use [default 100]')
    parser.add_argument('--debug', action='store_true', default=False, help='have print statements')
    parser.add_argument('--seed', type=int, default=100, help='how much of the query body to use [default 100]')
    args = parser.parse_args()
    # update args and print
    print "\nParameters:"
    for attr, value in sorted(args.__dict__.items()):
        print "\t{}={}".format(attr.upper(), value)

    print "Getting Embeddings..."
    embeddings, word_to_indx = ubuntu_data_utils.get_embeddings_tensor()

    if args.train:
        print "Getting Train Data..."
        train_data = AskUbuntuDataset.AskUbuntuDataset('train', word_to_indx, max_seq_length=args.len_query, training_data_size=args.training_data_size)
    print "Getting Dev Data..."
    dev_data = AskUbuntuDataset.AskUbuntuDataset('dev', word_to_indx, max_seq_length=args.len_query)
    print "Getting Test Data..."
    test_data = AskUbuntuDataset.AskUbuntuDataset('test', word_to_indx, max_seq_length=args.len_query)

    set_seeds(args)

    # model
    if args.snapshot is None:
        model = model_utils.get_model(embeddings, args)
    else :
        print '\nLoading model from [%s]...' % args.snapshot
        try:
            model = torch.load(args.snapshot)
        except :
            print "Sorry, This snapshot doesn't exist."
            exit()
    print model
    paramter_num = 0
    embedding_paramter_num = 0
    for param in model.parameters():
        if paramter_num == 0:
            embedding_paramter_num = np.prod(param.data.shape)
        paramter_num += np.prod(param.data.shape)
    print "Total number of parameters: {}".format(paramter_num)
    print "Number of trainable parameters: {}".format(paramter_num - embedding_paramter_num)

    # train
    if args.train:
        print "\nTraining..."
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        train_utils.train_model(train_data, dev_data, model, args)

    if args.eval:
        print "\nEvaluating on dev data:"
        evaluation_utils.evaluate_model(dev_data, model, args)
        print "\nEvaluating on test data:"
        evaluation_utils.evaluate_model(test_data, model, args)