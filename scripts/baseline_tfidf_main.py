import argparse
import sys
import os
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from datasets.TfIdfDataset import TfIdfDataset
from evaluation.Meter import AUCMeter
from evaluation.Evaluation import Evaluation
import utils.model_utils as model_utils
import torch
import numpy as np


def evaluate_model(dev_data, model, args):

    # TODO: change model name to baseline/tf-idf
    batch_size = len(dev_data)

    data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    all_sorted_labels = []
    auc = AUCMeter()

    for batch in data_loader:

        q_tfidf_tensors = batch['qid_tfidf_tensor']
        candidate_tfidf_tensors = torch.stack(batch['candidate_tfidf_tensors'])

        labels = torch.stack(batch['labels'],dim=1)
        labels = labels.numpy()

        cosine_similarities = model.compute(q_tfidf_tensors,
                                            candidate_tfidf_tensors)

        np_cosine_similarities = cosine_similarities.data.cpu().numpy()

        for i in xrange(labels.shape[0]):
            auc.add(np_cosine_similarities[i,:],labels[i,:])

        sorted_indices = np_cosine_similarities.argsort(axis=1)
        sorted_labels = labels[np.expand_dims(np.arange(sorted_indices.shape[0]),1), sorted_indices]
        sorted_labels = np.flip(sorted_labels,1)
        all_sorted_labels.append(sorted_labels)

    all_sorted_labels = np.concatenate(all_sorted_labels)
    evaluation = Evaluation(all_sorted_labels)

    print "MAP: {}".format(evaluation.get_MAP())
    print "MRR: {}".format(evaluation.get_MRR())
    print "Precision@1: {}".format(evaluation.get_precision(1))
    print "Precision@5: {}".format(evaluation.get_precision(5))
    print "AUC(): {}".format(auc.value())

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AskUbuntu Question Retrieval Network')
    # data loading
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
    parser.add_argument('--model_name', nargs="?", type=str, default='bow', help="should be baseline")
    parser.add_argument('--source', nargs="?", type=str, default='android', help="should be baseline")
    # other
    parser.add_argument('--debug', action='store_true', default=False, help='have print statements')

    args = parser.parse_args()
    # update args and print
    print "\nParameters:"
    for attr, value in sorted(args.__dict__.items()):
        print "\t{}={}".format(attr.upper(), value)

    print "Getting Dev Data..."
    dev_data = TfIdfDataset('dev', args.source)
    print "Getting Test Data..."
    test_data = TfIdfDataset('test', args.source)

    # model
    model = model_utils.get_model(None,args)

    print "\nEvaluating on dev data:"
    evaluate_model(dev_data, model, args)
    print "\nEvaluating on test data:"
    evaluate_model(test_data, model, args)