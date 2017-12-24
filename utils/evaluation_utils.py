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

    dev_dataset = dev_data.dataset

    data_for_evaluation = []
    auc = AUCMeter()

    q_title_tensors = [sample['q_title_tensor'] for sample in dev_dataset]
    q_body_tensors = [sample['q_body_tensor'] for sample in dev_dataset]

    positive_title_tensors = [
        torch.stack(sample['positive_title_tensors']) if len(sample['positive_title_tensors']) > 0 else np.array([]) for
        sample in dev_dataset]
    positive_body_tensors = [
        torch.stack(sample['positive_body_tensors']) if len(sample['positive_body_tensors']) > 0 else np.array([]) for
        sample in dev_dataset]
    negative_title_tensors = [
        torch.stack(sample['negative_title_tensors']) if len(sample['negative_title_tensors']) > 0 else np.array([]) for
        sample in dev_dataset]
    negative_body_tensors = [
        torch.stack(sample['negative_body_tensors']) if len(sample['negative_body_tensors']) > 0 else np.array([]) for
        sample in dev_dataset]

    for i in xrange(len(q_title_tensors)):
        q_title_tensor_i = autograd.Variable(q_title_tensors[i]).unsqueeze(0)
        q_body_tensor_i = autograd.Variable(q_body_tensors[i]).unsqueeze(0)
        if positive_title_tensors[i].shape[0]:
            # unsqueeze to make "batch size" of 1
            positive_title_tensors_i = autograd.Variable(positive_title_tensors[i]).unsqueeze(1)
            positive_body_tensors_i = autograd.Variable(positive_body_tensors[i]).unsqueeze(1)
        if negative_title_tensors[i].shape[0]:
            negative_title_tensors_i = autograd.Variable(negative_title_tensors[i]).unsqueeze(1)
            negative_body_tensors_i = autograd.Variable(negative_body_tensors[i]).unsqueeze(1)

        labels = np.array([1]*positive_title_tensors[i].shape[0] + [0]*negative_title_tensors[i].shape[0])

        if args.cuda:
            q_title_tensor_i = q_title_tensor_i.cuda()
            q_body_tensor_i = q_body_tensor_i.cuda()
            if positive_title_tensors[i].shape[0]:
                positive_title_tensors_i = positive_title_tensors_i.cuda()
                positive_body_tensors_i = positive_body_tensors_i.cuda()
            if negative_title_tensors[i].shape[0]:
                negative_title_tensors_i = negative_title_tensors_i.cuda()
                negative_body_tensors_i = negative_body_tensors_i.cuda()

        if positive_title_tensors[i].shape[0]:
            if args.model_name == "exploration":
                pos_cosine_sims, _ = model(q_title_tensor_i,
                                        q_body_tensor_i,
                                        positive_title_tensors_i,
                                        positive_body_tensors_i)
            else:
                pos_cosine_sims = model(q_title_tensor_i,
                                        q_body_tensor_i,
                                        positive_title_tensors_i,
                                        positive_body_tensors_i)
            pos_cosine_sims_np = pos_cosine_sims.data.cpu().numpy()
        else:
            pos_cosine_sims_np = np.array([[]])

        if negative_title_tensors[i].shape[0]:
            if args.model_name == "exploration":
                neg_cosine_sims, _ = model(q_title_tensor_i,
                                        q_body_tensor_i,
                                        negative_title_tensors_i,
                                        negative_body_tensors_i)
            else:
                neg_cosine_sims = model(q_title_tensor_i,
                                        q_body_tensor_i,
                                        negative_title_tensors_i,
                                        negative_body_tensors_i)
            neg_cosine_sims_np = neg_cosine_sims.data.cpu().numpy()
        else:
            neg_cosine_sims_np = np.array([[]])

        cosine_similarities = np.concatenate((pos_cosine_sims_np[0],neg_cosine_sims_np[0]))
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
