import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.misc_utils as misc_utils


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        return CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    elif args.model_name == 'dan':
        return DAN(embeddings, args)
    elif args.model_name == 'bow':
        return BOW(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


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
        q_title_embeddings = self.forward_helper(q_title_tensors)
        # if self.args.debug: misc_utils.print_shape_variable('q_title_embeddings', q_title_embeddings)
        q_body_embeddings = self.forward_helper(q_body_tensors)
        # if self.args.debug: misc_utils.print_shape_variable('q_body_embeddings', q_body_embeddings)
        q_output_before_mean = torch.stack([q_title_embeddings, q_body_embeddings])
        # if self.args.debug: misc_utils.print_shape_variable('q_output_before_mean', q_output_before_mean)
        q_output = torch.mean(q_output_before_mean, dim=0)
        # if self.args.debug: misc_utils.print_shape_variable('q_output', q_output)

        num_candidates, d2, d3 = candidate_title_tensors.size()

        title_embeddings = self.forward_helper(candidate_title_tensors.view(num_candidates*d2,d3))
        # if self.args.debug: misc_utils.print_shape_variable('title_embeddings', title_embeddings)
        body_embeddings = self.forward_helper(candidate_body_tensors.view(num_candidates*d2,d3))
        # if self.args.debug: misc_utils.print_shape_variable('body_embeddings', body_embeddings)
        candidate_outputs_before_mean = torch.stack([title_embeddings, body_embeddings])
        # if self.args.debug: misc_utils.print_shape_variable('candidate_outputs_before_mean', candidate_outputs_before_mean)
        candidate_outputs = torch.mean(candidate_outputs_before_mean, dim=0)
        # if self.args.debug: misc_utils.print_shape_variable('candidate_outputs', candidate_outputs)


        expanded_q_output0 = q_output.view(1,d2,self.hidden_dim)
        # if self.args.debug: misc_utils.print_shape_variable('expanded_q_output0', expanded_q_output0)
        expanded_q_output1 = expanded_q_output0.expand(num_candidates,d2,self.hidden_dim)
        # if self.args.debug: misc_utils.print_shape_variable('expanded_q_output1', expanded_q_output1)
        expanded_q_output2 = expanded_q_output1.contiguous().view(num_candidates*d2,self.hidden_dim)
        # if self.args.debug: misc_utils.print_shape_variable('expanded_q_output2', expanded_q_output2)

        expanded_cosine_similarities = F.cosine_similarity(expanded_q_output2, candidate_outputs, dim=1)
        # if self.args.debug: misc_utils.print_shape_variable('expanded_cosine_similarities', expanded_cosine_similarities)

        # TODO: Double check that this transformation is resizing the output as desired
        output0 = expanded_cosine_similarities.view(num_candidates,d2,1)
        # if self.args.debug: misc_utils.print_shape_variable('output0', output0)
        output1 = output0.view(num_candidates,d2).t() # .t() is transpose
        # if self.args.debug: misc_utils.print_shape_variable('output1', output1)

        return output1

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

    def __init__(self, embeddings, args):
        super(CNN, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape
        self.conv = nn.Conv1d(embed_dim, self.hidden_dim, 3, padding=2)
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
            lengths = lengths.expand(tensor.data.shape)
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
            raise Exception("Pooling method {} not implemented".format(self.pooling))
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
        mask = mask.permute(1,0,2)
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(x_perm, self.hidden)

        N, hd, co =  lstm_out.data.shape
        expanded_mask = mask.expand(N, hd, co)
        masked = torch.mul(expanded_mask,lstm_out)
        out = torch.sum(masked,0)

        return out

class BOW(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args):
        super(BOW, self).__init__(embeddings, args)
        self.vocab_size, embed_dim = embeddings.shape
        self.hidden_dim = self.vocab_size
        self.name = 'bow'

    def forward_helper(self, tensor):
        batch_size, seq_len = tensor.data.shape
        out = torch.Tensor(batch_size,self.vocab_size)
        df = torch.Tensor(1,self.vocab_size)
        for i in range(batch_size):
            sample = tensor[i].unsqueeze(0)
            mask = (sample != 0)
            length = torch.sum(mask,1).data[0]
            indices = sample.narrow(1,0,length)
            for j in range(length):
                df[0][indices[0][j].data[0]] += 1
        # for i in range(self.vocab_size):
        #     mask = (tensor == i)
        #     count = torch.sum(mask,1).data[0]
        #     df[0][i] = count
        for i in range(batch_size):
            sample = tensor[i].unsqueeze(0)
            mask = (sample != 0)
            length = torch.sum(mask,1).data[0]
            indices = sample.narrow(1,0,length)
            for j in range(length):
                # out[i][indices[0][j].data[0]] += 1*(1+np.log((1+batch_size)/(1+df)))
                index = indices[0][j].data[0]
                out[i][index] += 1*(1+np.log((1+batch_size)/(1+df[0][index])))
        return autograd.Variable(out)
