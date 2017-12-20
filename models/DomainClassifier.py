import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, args):
        super(DomainClassifier, self).__init__()

        self.args = args
        self.input_dim = args.hidden_dim  # the output dim of the encoders is the input dim to domain classifier
        self.hidden_1_dim = args.hidden_dim_dom1
        self.hidden_2_dim = args.hidden_dim_dom2

        self.hidden_1 = nn.Linear(self.input_dim, self.hidden_1_dim)
        self.relu_1 = nn.ReLU()
        self.hidden_2 = nn.Linear(self.hidden_1_dim, self.hidden_2_dim)
        self.relu_2 = nn.ReLU()
        self.hidden_3 = nn.Linear(self.hidden_2_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoding):
        z1 = self.hidden_1(encoding)
        h1 = self.relu_1(z1)
        z2 = self.hidden_2(h1)
        h2 = self.relu_2(z2)
        h3 = self.hidden_3(h2)
        out = self.sigmoid(h3)
        return out
