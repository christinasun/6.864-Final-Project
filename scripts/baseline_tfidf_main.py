import argparse
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from datasets.TfIdfDataset import TfIdfDataset
from evaluation.Meter import AUCMeter
from evaluation.Evaluation import Evaluation
import numpy as np
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity



def compute_baseline(dev_data):
    dev_dataset = dev_data.dataset
    N = len(dev_data)

    data_for_evaluation = []

    auc = AUCMeter()

    q_tfidf_tensors = [sample['qid_tfidf_tensor'] for sample in dev_dataset]
    positive_tfidf_tensors = [vstack(sample['positive_tfidf_tensors']) if len(sample['positive_tfidf_tensors']) > 0 else np.array([]) for sample in dev_dataset]
    negative_tfidf_tensors = [vstack(sample['negative_tfidf_tensors']) if len(sample['negative_tfidf_tensors']) > 0 else np.array([]) for sample in dev_dataset]

    for i in xrange(len(q_tfidf_tensors)):
        q_tfidf_tensor = q_tfidf_tensors[i]
        labels = np.array([1]*positive_tfidf_tensors[i].shape[0] + [0]*negative_tfidf_tensors[i].shape[0])
        pos_cosine_sims = cosine_similarity(q_tfidf_tensor,positive_tfidf_tensors[i])[0] if positive_tfidf_tensors[i].shape[0] else np.array([])
        neg_cosine_sims = cosine_similarity(q_tfidf_tensor,negative_tfidf_tensors[i])[0] if negative_tfidf_tensors[i].shape[0] else np.array([])
        cosine_similarities = np.concatenate((pos_cosine_sims,neg_cosine_sims))
        sorted_indices = cosine_similarities.argsort()
        sorted_labels = labels[sorted_indices]

        data_for_evaluation.append(np.flip(sorted_labels,0))
        auc.add(cosine_similarities, labels)

    evaluation = Evaluation(data_for_evaluation)

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