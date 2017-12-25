import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class AbstractEncoder(nn.Module):
    def __init__(self, embeddings, args):
        super(AbstractEncoder, self).__init__()
        self.args = args
        self.vocab_size, self.embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
        self.with_recon_loss = False
        self.pooling = 'mean'  # in future, we can make this configurable

    def get_mask(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        if self.pooling == 'mean':
            lengths = torch.sum(mask, 1)
            lengths = torch.clamp(lengths, 1, 100)  # avoids issues caused by zero-length sequences
            lengths = torch.unsqueeze(lengths, 1)
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask, lengths)
        return mask

    def pool(self, masked_sequences, dim):
        if self.pooling == 'mean':
            summed = torch.sum(masked_sequences, dim)
            output = summed
        elif self.pooling == 'max':
            themax = torch.max(masked_sequences, dim=dim)
            output = themax
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))
        return output

    def forward(self, tensor):
        pass

class AbstractEncoderWithReconstructionLoss(AbstractEncoder):
    # The main difference between AbstractEncoderWithReconstructionLoss and AbstractEncoder is that the in addition to
    # the encodings, AbstractEncoderWithReconstructionLoss also outputs the total reconstruction loss and the total
    # number of tokens passed in.
    def __init__(self, embeddings, args):
        super(AbstractEncoderWithReconstructionLoss, self).__init__(embeddings, args)
        self.with_recon_loss = True
        self.reconstructor = nn.Linear(self.hidden_dim, self.embed_dim)
        self.tanh = nn.Tanh()

    def get_total_number_of_words(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)
        lengths = torch.sum(mask, 1)
        lengths = torch.clamp(lengths, 1, 100)  # avoids issues caused by zero-length sequences
        return sum(lengths)



class CNN(AbstractEncoder):
    def __init__(self, embeddings, args, kernel_size=3):
        super(CNN, self).__init__(embeddings, args)
        self.name = 'cnn'
        self.encoder = nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size, padding=kernel_size - 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, tensor):
        mask = self.get_mask(tensor)

        x = self.embedding_layer(tensor)  # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0, 2, 1)
        hiddens = self.encoder(x_perm)
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:, :, :self.args.len_query]

        N, hd, co = tanh_x_cropped.data.shape
        mask = torch.unsqueeze(mask, 1)
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask, tanh_x_cropped)

        output = self.pool(masked, 2)
        return output


class LSTM(AbstractEncoder):
    def __init__(self, embeddings, args):
        super(LSTM, self).__init__(embeddings, args)
        self.name = 'lstm'
        self.hidden = self.init_hidden(self.args.batch_size)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, bidirectional=True, dropout=self.args.dropout)
        return

    def init_hidden(self, batch):
        h = autograd.Variable(torch.zeros(2, batch, self.hidden_dim // 2))
        c = autograd.Variable(torch.zeros(2, batch, self.hidden_dim // 2))
        if self.args.cuda:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    def forward(self, tensor):
        mask = self.get_mask(tensor)
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

        output = self.pool(masked, 0)
        return output


class CNNWithReconstructionLoss(AbstractEncoderWithReconstructionLoss):
    def __init__(self, embeddings, args, kernel_size=3):
        super(CNNWithReconstructionLoss, self).__init__(embeddings, args)
        self.name = 'cnn_with_recon_loss'
        self.encoder = nn.Conv1d(self.embed_dim, self.hidden_dim, kernel_size, padding=kernel_size - 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, tensor):
        mask = self.get_mask(tensor)
        total_number_of_words = self.get_total_number_of_words(tensor)

        x = self.embedding_layer(tensor)  # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0, 2, 1)
        hiddens = self.encoder(x_perm)
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:, :, :self.args.len_query]

        N, hd, co = tanh_x_cropped.data.shape
        mask = torch.unsqueeze(mask, 1)
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask, tanh_x_cropped)

        output = self.pool(masked, 2)

        hiddens = hiddens[:, :, :self.args.len_query]
        hiddens = hiddens.permute(0, 2, 1)
        x_hat = self.tanh(self.reconstructor(hiddens))
        total_reconstruction_loss = F.mse_loss(x_hat, x, size_average=False)

        return output, total_reconstruction_loss, total_number_of_words

class LSTMWithReconstructionLoss(AbstractEncoderWithReconstructionLoss):
    def __init__(self, embeddings, args):
        super(LSTMWithReconstructionLoss, self).__init__(embeddings, args)
        self.name = 'lstm_with_recon_loss'
        self.hidden = self.init_hidden(self.args.batch_size)
        self.encoder = nn.LSTM(self.embed_dim, self.hidden_dim // 2, bidirectional=True, dropout=self.args.dropout)
        return

    def init_hidden(self, batch):
        h = autograd.Variable(torch.zeros(2, batch, self.hidden_dim // 2))
        c = autograd.Variable(torch.zeros(2, batch, self.hidden_dim // 2))
        if self.args.cuda:
            h = h.cuda()
            c = c.cuda()
        return (h, c)

    def forward(self, tensor):
        mask = self.get_mask(tensor)
        mask = torch.unsqueeze(mask, 2)
        total_number_of_words = self.get_total_number_of_words(tensor)

        x = self.embedding_layer(tensor)
        batch_size = x.data.shape[0]
        x_perm = x.permute(1, 0, 2)
        mask = mask.permute(1, 0, 2)
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.encoder(x_perm, self.hidden)
        N, hd, co = lstm_out.data.shape
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask, lstm_out)

        output = self.pool(masked, 0)

        x_hat = self.tanh(self.reconstructor(lstm_out))
        total_reconstruction_loss = F.mse_loss(x_hat, x, size_average=False)

        return output, total_reconstruction_loss, total_number_of_words

