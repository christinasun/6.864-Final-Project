import argparse
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.ubuntu_data_utils as data_utils
import utils.train_utils as train_utils
import utils.model_utils as model_utils
import utils.evaluation_utils as evaluation_utils
import torch
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AskUbuntu Question Retrieval Network')
    # learning
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--dropout', type=float, default=0.2, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument('--num_negative', type=int, default=20, help='# negative examples for training [default: 20]')
    parser.add_argument('--training_data_size', type=int, default = 1000000, help='Number of training queries [default: 1000000]')
    # data loading
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
    # model
    parser.add_argument('--model_name', nargs="?", type=str, default='cnn', help="Form of model, i.e dan, rnn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=20, help='dimension of the hidden layer [default: 20]')
    parser.add_argument('--lstm_hidden_dim', type=int, default=240, help='dimension of the lstm hidden layer [default: 240]')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('--train', action='store_true', default=False, help='enable train')
    parser.add_argument('--eval', action='store_true', default=False, help='enable eval')
    # task
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--save_path', type=str, default="model.pt", help='Path where to dump model')
    # other
    parser.add_argument('--len_query', type=int, default=100, help='how much of the query body to use [default 100]')
    parser.add_argument('--debug', action='store_true', default=False, help='have print statements')
    args = parser.parse_args()
    # update args and print
    print "\nParameters:"
    for attr, value in sorted(args.__dict__.items()):
        print "\t{}={}".format(attr.upper(), value)

    print "Getting Embeddings..."
    embeddings, word_to_indx = data_utils.get_embeddings_tensor()
    print "Getting Train Data..."
    train_data = data_utils.AskUbuntuDataset('train', word_to_indx, max_length=args.len_query, trainning_data_size=args.training_data_size)
    print "Getting Test Data..."
    dev_data = data_utils.AskUbuntuDataset('dev', word_to_indx, max_length=args.len_query)

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

    print "\nTraining..."
    # train
    if args.train :
        train_utils.train_model(train_data, dev_data, model, args)

    if args.eval:
        evaluation_utils.evaluate_model(dev_data, model, args)
