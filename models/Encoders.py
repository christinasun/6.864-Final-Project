import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, embeddings, args, kernel_size=3):
        super(CNN, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
        self.conv = nn.Conv1d(embed_dim, self.hidden_dim, kernel_size, padding=kernel_size - 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.pooling = 'mean'
        self.name = 'cnn'

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        if self.pooling == 'mean':
            lengths = torch.sum(mask, 1)
            lengths = torch.clamp(lengths, 1, 100)
            lengths = torch.unsqueeze(lengths, 1)
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask, lengths)

        x = self.embedding_layer(tensor)  # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0, 2, 1)
        hiddens = self.conv(x_perm)
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:, :, :self.args.len_query]

        N, hd, co = tanh_x_cropped.data.shape
        mask = torch.unsqueeze(mask, 1)
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask, tanh_x_cropped)

        if self.pooling == 'mean':
            summed = torch.sum(masked, dim=2)
            output = summed
        elif self.pooling == 'max':
            themax = torch.max(masked, dim=2)
            output = themax
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))
        return output


class LSTM(nn.Module):
    def __init__(self, embeddings, args):
        super(LSTM, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
        self.hidden = self.init_hidden(self.args.batch_size)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim // 2, bidirectional=True, dropout=self.args.dropout)
        self.pooling = 'mean'
        self.name = 'lstm'
        return

    def init_hidden(self, batch):
        h = autograd.Variable(torch.zeros(2, batch, self.hidden_dim // 2))
        c = autograd.Variable(torch.zeros(2, batch, self.hidden_dim // 2))
        if self.args.cuda:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        if self.pooling == 'mean':
            lengths = torch.sum(mask, 1)
            lengths = torch.clamp(lengths, 1, 100)
            lengths = torch.unsqueeze(lengths, 1)
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask, lengths)
            mask = torch.unsqueeze(mask, 2)

        x = self.embedding_layer(tensor)
        batch_size = x.data.shape[0]
        x_perm = x.permute(1, 0, 2)
        mask = mask.permute(1, 0, 2)
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(x_perm, self.hidden)
        N, hd, co = lstm_out.data.shape
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask, lstm_out)

        if self.pooling == 'mean':
            summed = torch.sum(masked, 0)
            output = summed
        elif self.pooling == 'max':
            themax = torch.max(masked, dim=0)
            output = themax
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))
        return output


class CNN_all(nn.Module):
    def __init__(self, embeddings, args, kernel_size=3):
        super(CNN_all, self).__init__()
        self.args = args
        vocab_size, self.embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding(vocab_size, self.embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
        self.base = nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size, padding=kernel_size - 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.pooling = 'mean'
        self.name = 'cnn'

        self.reconstructor = nn.Linear(self.hidden_dim, self.embed_dim)

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        lengths = torch.sum(mask, 1)
        lengths = torch.clamp(lengths,1,100)
        total_num_words = sum(lengths)
        if self.pooling == 'mean':
            lengths = torch.unsqueeze(lengths, 1)
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask, lengths)

        x = self.embedding_layer(tensor)  # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0, 2, 1)
        hiddens = self.base(x_perm)
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:, :, :self.args.len_query]

        N, hd, co = tanh_x_cropped.data.shape
        mask = torch.unsqueeze(mask, 1)
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask, tanh_x_cropped)

        if self.pooling == 'mean':
            summed = torch.sum(masked, dim=2)
            output_pooled = summed
        elif self.pooling == 'max':
            themax = torch.max(masked, dim=2)
            output_pooled = themax
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))

        hiddens = hiddens[:, :, :self.args.len_query]
        hiddens = hiddens.permute(0,2,1)
        # print "hiddens"
        # print hiddens
        #hiddens1 = hiddens.contiguous().view(N,hd*co)
        # x_hat = self.tanh(self.lin(hiddens1)).view(N,self.embed_dim,co)
        # print "xhat"
        # x_hat
        x_hat = self.tanh(self.reconstructor(hiddens))
        # print "xhat_2"
        # print x_hat2

        # x_emb = autograd.Variable(x_perm.data, requires_grad=False)
        # print "x"
        # print x
        mse_loss = F.mse_loss(x_hat, x, size_average=False)
        return output_pooled, mse_loss, total_num_words*hd
        # return output_pooled, x_hat, x_perm
