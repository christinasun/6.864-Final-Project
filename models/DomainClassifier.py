import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class DomainClassifier(nn.Module):

    def __init__(self, args):
        super(DomainClassifier, self).__init__()

        self.args = args
        self.input_dim = args.hidden_dim
        self.hidden_1_dim = 300
        self.hidden_2_dim = 150

        self.hidden_1 = nn.Linear(self.input_dim, self.hidden_1_dim)
        self.hidden_2 = nn.Linear(self.hidden_1_dim, self.hidden_2_dim)
        self.softmax = nn.Softmax(self.hidden_2_dim,2)


    def forward(self, encoding):
        h1 = self.hidden_1(encoding)
        h2 = self.hidden_2(h1)
        out = self.softmax(h2)
        return out