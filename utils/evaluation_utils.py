import torch
import torch.autograd as autograd
import torch.utils.data as data
import numpy as np

def evaluate_model(dev_data, model, args):

    if args.cuda:
        model = model.cuda()

    model.eval()

    N = len(dev_data)
    data_loader = torch.utils.data.DataLoader(
        dev_data,
        batch_size=N,
        shuffle=False,
        drop_last=False)

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

        cosine_similarities = model(q_title_tensors, q_body_tensors, candidate_title_tensors, candidate_body_tensors)
        np_cosine_similarities = cosine_similarities.data.cpu().numpy()


        sorted_indices = np_cosine_similarities.argsort(axis=1)
        # print "labels shape: {}".format(labels.shape)
        # print "sorted_indices_shape: {}".format(sorted_indices.shape)

        sorted_labels = labels[np.expand_dims(np.arange(sorted_indices.shape[0]),1), sorted_indices]
        sorted_labels = np.flip(sorted_labels,1)
        evaluation = Evaluation(sorted_labels)
        print "Precision@5: {}".format(evaluation.get_precision(5))
        print "Precision@1: {}".format(evaluation.get_precision(1))
        print "MAP: {}".format(evaluation.get_MAP())
        print "MRR: {}".format(evaluation.get_MRR())
        return


# This was taken from the implementation found at https://github.com/taolei87/rcnn/tree/master/code/qa
class Evaluation():

    def __init__(self,data):
        self.data = data


    def get_precision(self,precision_at):
        scores = []
        for item in self.data:
            temp = item[:precision_at]
            rel = np.array([val==1 for val in item])
            if rel.any():
                scores.append(sum([1 if val==1 else 0 for val in temp])*1.0 / len(temp) if len(temp) > 0 else 0.0)
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0


    def get_MAP(self):
        scores = []
        missing_MAP = 0
        for item in self.data:
            temp = []
            count = 0.0
            for i,val in enumerate(item):
                if val == 1:
                    count += 1.0
                    temp.append(count/(i+1))
            if len(temp) > 0:
                scores.append(sum(temp) / len(temp))
            else:
                missing_MAP += 1
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0


    def get_MRR(self):

        scores = []
        for item in self.data:
            for i,val in enumerate(item):
                if val == 1:
                    scores.append(1.0/(i+1))
                    break

        return sum(scores)/len(scores) if len(scores) > 0 else 0.0