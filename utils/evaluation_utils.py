import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as data
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from evaluation.Meter import AUCMeter
from evaluation.Evaluation import Evaluation

def evaluate_model(dev_data, model, args):

    if args.cuda:
        model = model.cuda()

    model.eval()

    # TODO: change model name to baseline/tf-idf
    if args.model_name == 'bow':
        batch_size = len(dev_data) #reason: tfidf needs to be computer over the whole corpus
    else:
        batch_size = 10

    N = len(dev_data)
    data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    all_sorted_labels = []

    auc = AUCMeter()

    for batch in data_loader:

        q_title_tensors = autograd.Variable(batch['qid_title_tensor'])
        q_body_tensors = autograd.Variable(batch['qid_body_tensor'])

        candidate_title_tensors = autograd.Variable(torch.stack(batch['candidate_title_tensors']))
        candidate_body_tensors = autograd.Variable(torch.stack(batch['candidate_body_tensors']))

        labels = torch.stack(batch['labels'],dim=1)
        labels = labels.numpy()

        if args.cuda:
            q_title_tensors = q_title_tensors.cuda()
            q_body_tensors = q_body_tensors.cuda()
            candidate_title_tensors = candidate_title_tensors.cuda()
            candidate_body_tensors = candidate_body_tensors.cuda()

        cosine_similarities = model(q_title_tensors,
                                    q_body_tensors,
                                    candidate_title_tensors,
                                    candidate_body_tensors)
        np_cosine_similarities = cosine_similarities.data.cpu().numpy()

        for i in xrange(labels.shape[0]):
            auc.add(np_cosine_similarities[i,:],labels[i,:])

        sorted_indices = np_cosine_similarities.argsort(axis=1)
        # print "labels shape: {}".format(labels.shape)
        # print "sorted_indices_shape: {}".format(sorted_indices.shape)

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


