import torch
import torch.autograd as autograd
import torch.nn as nn


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
            lengths = torch.unsqueeze(lengths, 1)
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask, lengths)

        x = self.embedding_layer(tensor)  # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0, 2, 1)
        hiddens = self.conv(x_perm)
        # print("hiddens: ", hiddens.size())
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:, :, :self.args.len_query]

        N, hd, co = tanh_x_cropped.data.shape
        # print("tanh_x_cropped: ", tanh_x_cropped.size())
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
        # print("output: ", output.size())
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
        # print("lstm_out: ", lstm_out.size())
        # print("self.hidden: ", self.hidden[0].size())
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
        # print("output: ", output.size())
        return output

class CNN_recon(nn.Module):
    def __init__(self, embeddings, args, kernel_size=3):
        super(CNN_recon, self).__init__()
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

        self.tanh = nn.Tanh()
        self.lin = nn.Linear(self.hidden_dim*self.args.len_query, self.hidden_dim)

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        if self.pooling == 'mean':
            lengths = torch.sum(mask, 1)
            lengths = torch.unsqueeze(lengths, 1)
            lengths_only = lengths
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask, lengths)

        x = self.embedding_layer(tensor)  # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0, 2, 1)
        hiddens = self.conv(x_perm)
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:, :, :self.args.len_query]
        N, hd, co = tanh_x_cropped.data.shape

        hiddens = hiddens[:, :, :self.args.len_query]
        hiddens = hiddens.contiguous().view(N,hd*co)
        x_hat = self.tanh(self.lin(hiddens))

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
        recon_loss = torch.norm(x_hat.sub(output), 2, 1)
        recon_loss = torch.unsqueeze(recon_loss, 1)
        recon_loss = torch.div(recon_loss, lengths_only)
        return recon_loss