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
        self.lstm_hidden_dim = args.lstm_hidden_dim
        return

    def forward(self, q_title_tensors, q_body_tensors, candidate_title_tensors, candidate_body_tensors):
        q_title_embeddings = self.forward_helper(q_title_tensors,10 * np.ones(self.args.batch_size,dtype=np.long))
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

    def forward_helper(self, tensor, lengths):
        print lengths
        if self.args.debug: misc_utils.print_shape_variable('tensor',tensor)
        x = self.embedding_layer(tensor) # (batch size, width (length of text), height (embedding dim))
        if self.args.debug: misc_utils.print_shape_variable('x',x)
        x_perm = x.permute(0,2,1)
        if self.args.debug: misc_utils.print_shape_variable('x_perm',x_perm)
        hiddens = self.conv(x_perm)
        if self.args.debug: misc_utils.print_shape_variable('hiddens',hiddens)
        post_dropout = self.dropout(hiddens)
        if self.args.debug: misc_utils.print_shape_variable('post_dropout', post_dropout)
        tanh_x = self.tanh(post_dropout)
        if self.args.debug: misc_utils.print_shape_variable('tanh_x', tanh_x)
        if self.args.debug: print tanh_x

        N, hd, co =  tanh_x.data.shape

        mask = torch.zeros(N,hd,co)
        if self.args.debug: misc_utils.print_shape_tensor('mask', mask)

        if self.pooling == 'mean':
            for i in xrange(N):
                mask[i,:,0:lengths[i]] = 1.0/lengths[i]
            mask = autograd.Variable(mask,requires_grad=False)
            if self.args.cuda:
                mask.cuda()
            if self.args.debug: misc_utils.print_shape_variable('mask', mask)
            masked = torch.mul(mask,tanh_x)
            if self.args.debug: misc_utils.print_shape_variable('masked', masked)
            summed = torch.sum(masked,dim=2)
            if self.args.debug: misc_utils.print_shape_variable('summed', summed)
            output = summed
        else:
            raise Exception("Pooling method {} not implemented".format(self.pooling))

        return output


class LSTM(AbstractAskUbuntuModel):

    # def __init__(self, embeddings, args):
    #     super(LSTM, self).__init__(embeddings, args)
    #     vocab_size, embed_dim = embeddings.shape
    #     self.num_layers = 1
    #     self.hidden = self.init_hidden()
    #     self.lstm = nn.LSTM(embed_dim, self.lstm_hidden_dim)
    #     self.W_o = nn.Linear(embed_dim*self.lstm_hidden_dim/2, self.hidden_dim)

    # def init_hidden(self):
    #     return (autograd.Variable(torch.zeros(self.num_layers, 1, self.lstm_hidden_dim)),
    #             autograd.Variable(torch.zeros(self.num_layers, 1, self.lstm_hidden_dim)))

    # def forward_helper(self, tensor):
    #     x = self.embedding_layer(tensor)
    #     lstm_out, self.hidden = self.lstm(x, self.hidden)
    #     flattened = lstm_out.view(len(tensor), 1, -1).squeeze(1)
    #     out = self.W_o(flattened)
    #     return out



    # def __init__(self, embeddings, args):
    #     super(LSTM, self).__init__(embeddings, args)
    #     vocab_size, embed_dim = embeddings.shape
    #     len_query = args.len_query
    #     self.num_layers = 1
    #     self.lstm = nn.LSTMCell(200, 240)

    # def init_hidden(self):
    #     return (autograd.Variable(torch.zeros(100, 240)))

    # def init_c(self):
    #     return (autograd.Variable(torch.zeros(100, 240)))

    # def forward_helper(self, tensor):
    #     output = []
    #     x = self.embedding_layer(tensor)
    #     hx = autograd.Variable(torch.zeros(100, 240))
    #     cx = autograd.Variable(torch.zeros(100, 240))
    #     for i in range(len(tensor)):
    #         hx, cx = self.lstm(x[i], (hx, cx))
    #         output.append(hx)
    #     out = torch.mean(torch.stack(output),0)
    #     print("out: ", out.size())
    #     return out

    def __init__(self, embeddings, args):
        super(LSTM, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape # embed_dim = 200
        self.len_query = args.len_query
        self.num_layers = 1
        self.lstm = nn.LSTMCell(embed_dim, self.lstm_hidden_dim)
        self.W_o = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)

    def forward_helper(self, tensor):
        x = self.embedding_layer(tensor)
        hx = autograd.Variable(torch.zeros(1, self.lstm_hidden_dim))
        cx = autograd.Variable(torch.zeros(1, self.lstm_hidden_dim))
        if self.args.cuda:
            hx = hx.cuda()
            cx = cx.cuda()
        batch_output = []
        for i in range(len(tensor)):
            seq = x[i]
            output = []
            for token in range(self.len_query):
                hx, cx = self.lstm(seq[token], (hx, cx))
                output.append(hx)
            out = torch.mean(torch.stack(output),0)
            batch_output.append(out)
        total = torch.stack(batch_output)
        final_output = self.W_o(total).squeeze(1)
        return final_output

