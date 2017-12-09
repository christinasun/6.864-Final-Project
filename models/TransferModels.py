import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class AbstractTransferModel(nn.Module):

    def __init__(self, embeddings, args):
        super(AbstractTransferModel, self).__init__()

        self.name = None
        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False

        self.encoder_hidden_dim = args.hidden_dim

        return

    def forward(self,
                q_title_tensors,
                q_body_tensors,
                candidate_title_tensors,
                candidate_body_tensors,
                for_dc_title_tensors,
                for_dc_body_tensors):


        ### LABEL PREDICTION (lp) ###
        q_title_encodings = self.forward_helper(q_title_tensors)
        q_body_encodings = self.forward_helper(q_body_tensors)
        q_encoding_before_mean = torch.stack([q_title_encodings, q_body_encodings])
        q_encoding = torch.mean(q_encoding_before_mean, dim=0)

        # get the encodings for the flattened out candidate tensors
        num_candidates, batch_size, embedding_dim = candidate_title_tensors.size()
        candidate_title_encodings = self.encode(candidate_title_tensors.view(num_candidates*batch_size,embedding_dim))
        candidate_body_encodings = self.encode(candidate_body_tensors.view(num_candidates*batch_size,embedding_dim))
        candidate_encodings_before_mean = torch.stack([candidate_title_encodings, candidate_body_encodings])
        candidate_encodings = torch.mean(candidate_encodings_before_mean, dim=0)

        # expand and flatten q encodings so that we can pass it through cosine similarities with flattened out
        # candidate encodings
        expanded_q_encoding = q_encoding.view(1,batch_size,self.hidden_dim)
        expanded_q_encoding = expanded_q_encoding.expand(num_candidates,batch_size,self.hidden_dim)
        expanded_q_encoding = expanded_q_encoding.contiguous().view(num_candidates*batch_size,self.hidden_dim)

        # compute cosine similarities
        expanded_cosine_similarities = F.cosine_similarity(expanded_q_encoding, candidate_encodings, dim=1)

        # un-flatten cosine similarities so that we get output of size batch_size * num_candidates (t() is transpose)
        lp_output = expanded_cosine_similarities.view(num_candidates,batch_size,1).view(num_candidates,batch_size).t()

        ### DOMAIN CLASSIFICATION (dc) ###

        # get the encodings for the flattened out domain classifier tensors
        num_for_dc, batch_size, embedding_dim = for_dc_title_tensors.size()
        for_dc_title_encodings = self.encode(for_dc_title_tensors.view(num_for_dc * batch_size, embedding_dim))
        for_dc_body_encodings = self.encode(for_dc_title_tensors.view(num_for_dc * batch_size, embedding_dim))
        for_dc_encodings_before_mean = torch.stack([for_dc_title_encodings, for_dc_body_encodings])
        for_dc_encodings = torch.mean(for_dc_encodings_before_mean, dim=0)

        expanded_labels = self.classify_domain(for_dc_encodings)

        dc_output = expanded_labels.view(num_candidates,batch_size,1).view(num_candidates,batch_size).t()

        return lp_output, dc_output

    def encode(self, tensor):
        pass

    def classify_domain(self, tensor):
        pass


class DAN(AbstractTransferModel):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape
        self.W_hidden = nn.Linear(embed_dim, embed_dim)
        self.W_out = nn.Linear(embed_dim, self.hidden_dim)
        self.name = 'dan'

    def encode(self, tensor):
        all_x = self.embedding_layer(tensor)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out

    def classify_domain(self, tensor):
        return 1