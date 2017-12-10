import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class LabelPredictor(nn.Module):
    # The label predictor abstracts away the encoder.
    # It takes in a title/body pairs representing a query and a list of title/body pairs representing the candidates for
    # each query. It encodes each query and its candidates using the encoder. Then, it computes the cosine similarities between each
    # query and all of its candidates. It outputs these cosine similarities "labels".

    def __init__(self, embeddings, args, encoder, domain_classifier):
        super(LabelPredictor, self).__init__()

        self.args = args
        self.encoder = encoder
        self.hidden_dim = args.hidden_dim

        return

    def forward(self,
                q_title_tensors,
                q_body_tensors,
                candidate_title_tensors,
                candidate_body_tensors):

        q_title_encodings = self.encoder(q_title_tensors)
        q_body_encodings = self.encoder(q_body_tensors)
        q_encoding_before_mean = torch.stack([q_title_encodings, q_body_encodings])
        q_encoding = torch.mean(q_encoding_before_mean, dim=0)

        # get the encodings for the flattened out candidate tensors
        num_candidates, batch_size, embedding_dim = candidate_title_tensors.size()
        candidate_title_encodings = self.encoder(candidate_title_tensors.view(num_candidates*batch_size,embedding_dim))
        candidate_body_encodings = self.encoder(candidate_body_tensors.view(num_candidates*batch_size,embedding_dim))
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

        return lp_output