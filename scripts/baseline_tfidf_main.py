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
from scipy.sparse import coo_matrix, vstack, hstack
from sklearn.metrics.pairwise import cosine_similarity



def evaluate_model(dev_data, model, args):

    dev_dataset = dev_data.dataset
    batch_size = len(dev_data)
    print batch_size


    data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    all_sorted_labels = []
    auc = AUCMeter()

    sample0 = dev_dataset[0]
    num_candidates = 101
    print "sample 0"
    print sample0['qid_tfidf_tensor']
    print sample0['qid_tfidf_tensor'].shape
    print vstack(sample0['candidate_tfidf_tensors']).shape

    q_tfidf_tensors = vstack([vstack([sample['qid_tfidf_tensor']]*num_candidates) for sample in dev_dataset])
    q_tfidf_tensors = vstack(q_tfidf_tensors)
    print "q_tfidf_tensors shape"
    print q_tfidf_tensors.shape

    a = vstack(sample['candidate_tfidf_tensors'])
    print a.shape
    candidate_tfidf_tensors = vstack([vstack(sample['candidate_tfidf_tensors'][0:num_candidates]) for sample in dev_dataset])
    print "candidate_tfidf_tensors"
    print candidate_tfidf_tensors.shape

    labels = [sample['labels'] for sample in dev_dataset]
    labels = np.array(labels)
    print "labels"
    print labels

    # cosine_similarities = model.compute(q_tfidf_tensors,
    #                                     candidate_tfidf_tensors)
    expanded_cosine_similarities = cosine_similarity(q_tfidf_tensors, candidate_tfidf_tensors)
    print "expanded"
    print expanded_cosine_similarities.shape

    # un-flatten cosine similarities so that we get output of size batch_size * num_candidates (t() is transpose)
    output = expanded_cosine_similarities.view(num_candidates, batch_size, 1).view(num_candidates, batch_size).t()


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
    print "AUC(): {}".format(auc.value(max_fpr=0.05))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AskUbuntu Question Retrieval Network')
    # data loading
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='num workers for data loader')
    parser.add_argument('--model_name', nargs="?", type=str, default='baseline', help="should be baseline")
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