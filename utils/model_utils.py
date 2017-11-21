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
    if args.model_name == 'lstm':
        return DAN(embeddings, args)
    elif args.model_name == 'cnn':
        return RNN(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


class CNN(nn.Module):

    def __init__(self, embeddings, args):
        # super(DAN, self).__init__()
        #
        # self.args = args
        # vocab_size, embed_dim = embeddings.shape
        #
        # self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        # self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        #
        # self.W_hidden = nn.Linear(embed_dim, 200)
        # self.W_out = nn.Linear(200, 1)
        pass


    def forward(self, x_indx):
        # all_x = self.embedding_layer(x_indx)
        # avg_x = torch.mean(all_x, dim=1)
        # hidden = F.relu( self.W_hidden(avg_x) )
        # out = self.W_out(hidden)
        # return out
        pass


class LSTM(nn.Module):
    # TODO: Implement

    def __init__(self):
        pass

    def forward(self):
        pass
