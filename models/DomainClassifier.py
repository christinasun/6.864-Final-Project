import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, args):
        super(DomainClassifier, self).__init__()

        self.args = args
        self.input_dim = args.hidden_dim  # the output dim of the encoders is the input dim to domain classifier
        self.hidden_1_dim = args.hidden_dim_dom1
        self.hidden_2_dim = args.hidden_dim_dom2

        self.hidden_1 = nn.Linear(self.input_dim, self.hidden_1_dim)
        self.hidden_2 = nn.Linear(self.hidden_1_dim, self.hidden_2_dim)
        self.hidden_3 = nn.Linear(self.hidden_2_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoding):
        d0 = self.dropout(encoding)
        z1 = self.hidden_1(d0)
        d1 = self.dropout(z1)
        h1 = self.relu(d1)
        z2 = self.hidden_2(h1)
        d2 = self.dropout(z2)
        h2 = self.relu(d2)
        h3 = self.hidden_3(h2)
        out = self.sigmoid(h3)
        return out
