import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
        self.hidden = self.init_hidden(self.args.batch_size)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim//2, bidirectional=True, dropout=self.args.dropout)
        self.pooling = 'mean'
        self.name = 'lstm'
        return

    def init_hidden(self, batch):
        h = autograd.Variable(torch.zeros(2, batch, self.hidden_dim//2))
        c = autograd.Variable(torch.zeros(2, batch, self.hidden_dim//2))
        if self.args.cuda:
            h = h.cuda()
            c = c.cuda()
        return (h,c)

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        if self.pooling == 'mean':
            lengths = torch.sum(mask,1)
            lengths = torch.unsqueeze(lengths,1)
            lengths = lengths.expand(tensor.data.shape)
            mask = torch.div(mask,lengths)
            mask = torch.unsqueeze(mask,2)

        x = self.embedding_layer(tensor)
        batch_size = x.data.shape[0]
        x_perm = x.permute(1,0,2)
        mask = mask.permute(1,0,2)
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(x_perm, self.hidden)

        N, hd, co =  lstm_out.data.shape
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask,lstm_out)

        if self.pooling == 'mean':
            summed = torch.sum(masked,0)
            output = summed
        elif self.pooling == 'max':
            themax = torch.max(masked,dim=0)
            output = themax
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))
        return output