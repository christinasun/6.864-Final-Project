import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# TODO: Do we want to add dropout to the lstm?
class AbstractAskUbuntuModel(nn.Module):

    def __init__(self, embeddings, args):
        super(AbstractAskUbuntuModel, self).__init__()

        self.args = args

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
        self.name = None
        return

    def forward(self,
                q_title_tensors,
                q_body_tensors,
                candidate_title_tensors,
                candidate_body_tensors):

        q_title_encodings = self.forward_helper(q_title_tensors)
        q_body_encodings = self.forward_helper(q_body_tensors)
        q_encoding_before_mean = torch.stack([q_title_encodings, q_body_encodings])
        q_encoding = torch.mean(q_encoding_before_mean, dim=0)

        num_candidates, batch_size, embedding_dim = candidate_title_tensors.size()

        # get the encodings for the flattened out candidate tensors
        title_encodings = self.forward_helper(candidate_title_tensors.view(num_candidates*batch_size,embedding_dim))
        body_encodings = self.forward_helper(candidate_body_tensors.view(num_candidates*batch_size,embedding_dim))
        candidate_encodings_before_mean = torch.stack([title_encodings, body_encodings])
        candidate_encodings = torch.mean(candidate_encodings_before_mean, dim=0)

        # expand and flatten q encodings so that we can pass it through cosine similarities with flattened out
        # candidate encodings
        expanded_q_encoding = q_encoding.view(1,batch_size,self.hidden_dim)
        expanded_q_encoding = expanded_q_encoding.expand(num_candidates,batch_size,self.hidden_dim)
        expanded_q_encoding = expanded_q_encoding.contiguous().view(num_candidates*batch_size,self.hidden_dim)

        # compute cosine similarities
        expanded_cosine_similarities = F.cosine_similarity(expanded_q_encoding, candidate_encodings, dim=1)

        # un-flatten cosine similarities so that we get output of size batch_size * num_candidates (t() is transpose)
        output = expanded_cosine_similarities.view(num_candidates,batch_size,1).view(num_candidates,batch_size).t()

        return output

    def forward_helper(self, tensor):
        pass


class DAN(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape
        self.W_hidden = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, self.hidden_dim)
        self.name = 'dan'

    def forward_helper(self, tensor):
        all_x = self.embedding_layer(tensor)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out


class CNN(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args, kernel_size=3):
        super(CNN, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape
        self.conv = nn.Conv1d(embed_dim, self.hidden_dim, kernel_size, padding=kernel_size-1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.pooling = 'mean'
        self.name = 'cnn'

    def forward_helper(self, tensor):
        mask = (tensor != 0)
        if self.args.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        if self.pooling == 'mean':
            lengths = torch.sum(mask,1)
            lengths = torch.unsqueeze(lengths,1)
            lengths = lengths.expand(mask.data.shape)
            mask = torch.div(mask,lengths)

        x = self.embedding_layer(tensor) # (batch size, width (length of text), height (embedding dim))
        x_perm = x.permute(0,2,1)
        hiddens = self.conv(x_perm)
        post_dropout = self.dropout(hiddens)
        tanh_x = self.tanh(post_dropout)
        tanh_x_cropped = tanh_x[:,:,:self.args.len_query]

        N, hd, co =  tanh_x_cropped.data.shape
        mask = torch.unsqueeze(mask,1)
        expanded_mask = mask.expand(N, hd, co)

        masked = torch.mul(expanded_mask,tanh_x_cropped)

        if self.pooling == 'mean':
            summed = torch.sum(masked, dim=2)
            output = summed
        elif self.pooling == 'max':
            themax = torch.max(masked, dim=2)
            output = themax
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))
        return output


class LSTM(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args):
        super(LSTM, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape
        self.num_layers = 1
        self.hidden = self.init_hidden(self.args.batch_size)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim//2, bidirectional=True)
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.pooling = 'mean'
        self.name = 'lstm'

    def init_hidden(self, batch):
        h = autograd.Variable(torch.zeros(2, batch, self.hidden_dim//2))
        c = autograd.Variable(torch.zeros(2, batch, self.hidden_dim//2))
        if self.args.cuda:
            h = h.cuda()
            c = c.cuda()
        return (h,c)

    def forward_helper(self, tensor):
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
        x_perm = self.dropout(x_perm)
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