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


    def forward(self, q_tensors, candidate_tensors):
        q_title_embedding = self.forward_helper(q_tensors[0])
        q_body_embedding = self.forward_helper(q_tensors[1])
        print "q_title_embedding: {}".format(q_title_embedding)
        print "q_body_embedding: {}".format(q_body_embedding)


        q_embedding = torch.cat((q_title_embedding, q_body_embedding), 0)
        q_embedding = torch.mean(q_embedding, dim=2)
        print "q_embedding: {}".format(q_embedding)

        title_tensors, body_tensors = zip(*candidate_tensors)
        print "title tensors: {}",format(title_tensors)
        print "bodytensors: {}",format(body_tensors)
        title_embeddings = self.forward_helper(title_tensors)
        body_embeddings = self.forward_helper(body_tensors)
        print "title embeddings: {}",format(title_embeddings)
        print "body embeddings: {}",format(body_embeddings)
        return "we did it"

    def forward_helper(self, tensor):
        all_x = self.embedding_layer(x_indx)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out


class LSTM(nn.Module):
    # TODO: Implement

    def __init__(self):
        pass

    def forward(self):
        pass
