import argparse
import sys
import os
import torch
from os.path import dirname, realpath
import torch

sys.path.append(dirname(dirname(realpath(__file__))))
import utils.android_data_utils as android_data_utils
import utils.exploration_train_utils as train_utils
import utils.model_utils as model_utils
import utils.evaluation_utils as evaluation_utils
from utils.misc_utils import set_seeds

from datasets.AskUbuntuDataset import AskUbuntuDataset
from datasets.AndroidDataset import AndroidDataset
from datasets.TransferDatasetGenerator import TransferDatasetGenerator
import numpy as np
from models.LabelPredictor import LabelPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AskUbuntu Question Retrieval Network')
    # learning
    parser.add_argument('--encoder_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--domain_classifier_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--domain_classifier_lam', type=float, default=0.001, help='lambda [default: 0.001]')
    parser.add_argument('--reconstruction_lam', type=float, default=0.001, help='lambda [default: 0.001]')
    parser.add_argument('--dropout', type=float, default=0.2, help='initial learning rate [default: 0.001]')
    parser.add_argument('--margin', type=float, default=0.5, help='margin size [default: 0.5]')
    parser.add_argument('--epochs', type=int, default=6, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size for training [default: 128]')
    parser.add_argument('--num_negative', type=int, default=20, help='# negative examples for training [default: 20]')
    parser.add_argument('--training_data_size', type=int, default = 1000000, help='Number of training queries [default: 1000000]')
    # data loading
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
    # model
    parser.add_argument('--model_name', nargs="?", type=str, default='exploration', help="Form of model, i.e dan, rnn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=20, help='dimension of the hidden layer [default: 20]')
    parser.add_argument('--hidden_dim_dom1', type=int, default=300, help='dimension of the hidden layer [default: 300]')
    parser.add_argument('--hidden_dim_dom2', type=int, default=150, help='dimension of the hidden layer [default: 150]')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')
    parser.add_argument('--train', action='store_true', default=False, help='enable train')
    parser.add_argument('--eval', action='store_true', default=False, help='enable eval')
    # task
    parser.add_argument('--encoder_snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--domain_classifier_snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
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
    embeddings, word_to_indx = android_data_utils.get_embeddings_tensor()

    if args.train:
        print "Getting Train Data..."
        label_predictor_train_data = AskUbuntuDataset('train', word_to_indx, max_seq_length=args.len_query, training_data_size=args.training_data_size)
        adversary_train_data_generator = TransferDatasetGenerator('train', word_to_indx, max_seq_length=args.len_query)

    print "Getting Ubuntu Dev Data..."
    ubuntu_dev_data = AskUbuntuDataset('dev', word_to_indx, max_seq_length=args.len_query)
    print "Getting Ubuntu Test Data..."
    ubuntu_test_data = AskUbuntuDataset('test', word_to_indx, max_seq_length=args.len_query)
    print "Getting Android Dev Data..."
    android_dev_data = AndroidDataset('dev', word_to_indx, max_seq_length=args.len_query)
    print "Getting Android Test Data..."
    android_test_data = AndroidDataset('test', word_to_indx, max_seq_length=args.len_query)

    set_seeds(args)

    # model
    if args.encoder_snapshot is None or args.domain_classifier_snapshot is None:
        encoder_model, domain_classifier_model = model_utils.get_model(embeddings, args)

    else :
        print 'Loading models from {} and {} ...'.format(args.encoder_snapshot, args.domain_classifier_snapshot)
        try:
            encoder_model = torch.load(args.encoder_snapshot)
            domain_classifier_model = torch.load(args.domain_classifier_snapshot)
        except :
            print "Sorry, This snapshot doesn't exist."
            exit()

    print encoder_model
    parameter_num = 0
    embedding_parameter_num = 0
    for param in encoder_model.parameters():
        if parameter_num == 0:
            embedding_parameter_num = np.prod(param.data.shape)
        parameter_num += np.prod(param.data.shape)
    print "Total number of encoder parameters: {}".format(parameter_num)
    print "Number of trainable encoder parameters: {}".format(parameter_num - embedding_parameter_num)

    print domain_classifier_model
    parameter_num = 0
    for param in domain_classifier_model.parameters():
        parameter_num += np.prod(param.data.shape)
    print "Total number of domain classifier parameters: {}".format(parameter_num)

    # train
    if args.train:
        print "\nTraining..."
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        train_utils.train_model(label_predictor_train_data, adversary_train_data_generator, android_dev_data, encoder_model, domain_classifier_model, args)

    if args.eval:
        label_predictor = LabelPredictor(encoder_model)

        print "\nEvaluating on android dev data:"
        evaluation_utils.evaluate_model(android_dev_data, label_predictor, args)
        print "\nEvaluating on android test data:"
        evaluation_utils.evaluate_model(android_test_data, label_predictor, args)

        print "\nEvaluating on ubuntu dev data:"
        evaluation_utils.evaluate_model(ubuntu_dev_data, label_predictor, args)
        print "\nEvaluating on ubuntu test data:"
        evaluation_utils.evaluate_model(ubuntu_test_data, label_predictor, args)