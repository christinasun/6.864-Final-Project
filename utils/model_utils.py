import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import tqdm
import datetime
import pdb


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        return CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))



class CNN(nn.Module):

    def __init__(self, embeddings, args):
        super(CNN, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.W_hidden = nn.Linear(embed_dim, 200)
        self.W_out = nn.Linear(200, 200)
        pass


    def forward(self, q_title_tensors, q_body_tensors, candidate_title_tensors, candidate_body_tensors):
        q_title_embeddings = self.forward_helper(q_title_tensors)
        q_body_embeddings = self.forward_helper(q_body_tensors)
        print "\n\nq_title_embedding: {}".format(q_title_embeddings)
        print "\n\nq_body_embedding: {}".format(q_body_embeddings)

        q_embedding_before_mean = torch.stack([q_title_embeddings, q_body_embeddings])
        print "q_embedding_before_mean: {}".format(q_embedding_before_mean.size())

        q_embedding = torch.mean(q_embedding_before_mean, dim=0)
        print "q_embedding: {}".format(q_embedding.size())

        num_candidates, d2, d3 = candidate_title_tensors.size()

        print "title tensors: {}",format(candidate_title_tensors.size())
        print "body tensors: {}",format(candidate_body_tensors.size())
        title_embeddings = self.forward_helper(candidate_title_tensors.view(num_candidates*d2,d3))
        body_embeddings = self.forward_helper(candidate_body_tensors.view(num_candidates*d2,d3))
        candidate_embeddings_before_mean = torch.stack([title_embeddings, body_embeddings])
        candidate_embeddings = torch.mean(candidate_embeddings_before_mean, dim=0)

        print "candidate_embeddings_before_mean: {}".format(candidate_embeddings_before_mean.size())
        print "candidate_embeddings: {}".format(candidate_embeddings.size())

        expanded_q_embedding = q_embedding.view(1,d2,d3).expand(num_candidates,d2,d3).contiguous().view(num_candidates*d2,d3)
        print "expanded_q_embedding: {}".format(expanded_q_embedding.size())

        expanded_cosine_similarites = F.cosine_similarity(expanded_q_embedding, candidate_embeddings, dim=1)
        print "expanded_cosine_similarites: {}".format(expanded_cosine_similarites.size())

        output = expanded_cosine_similarites.view(num_candidates,d2,1).view(num_candidates,d2).t()
        print "output: {}".format(output.size())

        return output

    def forward_helper(self, tensor):
        all_x = self.embedding_layer(tensor)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out

    def cosine_similarity(self, q_embeddings, candidate_embeddings):
        pass
        


class LSTM(nn.Module):
    # TODO: Implement

    def __init__(self):
        pass

    def forward(self):
        pass
