import argparse
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from datasets.TfIdfDataset import TfIdfDataset
from evaluation.Meter import AUCMeter
from evaluation.Evaluation import Evaluation
import utils.model_utils as model_utils
import numpy as np
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity



def compute_baseline(dev_data):
    dev_dataset = dev_data.dataset
    N = len(dev_data)

    all_sorted_labels = []

    num_candidates = len(dev_dataset[0][candidates])

    q_tfidf_tensors = vstack([vstack([sample['qid_tfidf_tensor']]*num_candidates) for sample in dev_dataset])
    candidate_tfidf_tensors = vstack([vstack(sample['candidate_tfidf_tensors'][0:num_candidates]) for sample in dev_dataset])
    labels = np.array([sample['labels'][0:num_candidates] for sample in dev_dataset])

    cosine_similarities = np.zeros(N*num_candidates)
    for i in xrange(q_tfidf_tensors.shape[0]):
        cosine_similarities[i] = cosine_similarity(q_tfidf_tensors[i], candidate_tfidf_tensors[i])

    cosine_similarities_reshaped = cosine_similarities.reshape(N, num_candidates)

    sorted_indices = cosine_similarities_reshaped.argsort(axis=1)
    sorted_labels = labels[np.expand_dims(np.arange(sorted_indices.shape[0]),1), sorted_indices]
    sorted_labels = np.flip(sorted_labels,1)
    all_sorted_labels.append(sorted_labels)

    all_sorted_labels = np.concatenate(all_sorted_labels)
    evaluation = Evaluation(all_sorted_labels)

    auc = AUCMeter()
    for i in xrange(N):
        auc.add(cosine_similarities_reshaped[i, :], labels[i, :])

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

    print "\nEvaluating on dev data:"
    compute_baseline(dev_data)
    print "\nEvaluating on test data:"
    compute_baseline(test_data)