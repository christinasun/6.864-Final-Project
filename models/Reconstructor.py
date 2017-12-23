import torch
import torch.nn.functional as F
import torch.nn as nn

class Reconstructor(nn.Module):
    def __init__(self, encoder, reconstructor):
        super(Reconstructor, self).__init__()
        self.args = encoder.args
        self.encoder = encoder
        self.reconstructor = reconstructor
        self.hidden_dim = self.args.hidden_dim
        return

    def forward(self,
                q_title_tensors,
                q_body_tensors,
                candidate_title_tensors,
                candidate_body_tensors):
        q_title_encodings = self.encoder(q_title_tensors)
        q_body_encodings = self.encoder(q_body_tensors)
        q_encoding_before_mean = torch.cat([q_title_encodings, q_body_encodings])
        # q_encoding = torch.mean(q_encoding_before_mean, dim=0)

        # get the encodings for the flattened out candidate tensors
        num_candidates, batch_size, embedding_dim = candidate_title_tensors.size()
        candidate_title_encodings = self.encoder(candidate_title_tensors.view(num_candidates * batch_size, embedding_dim))
        candidate_body_encodings = self.encoder(candidate_body_tensors.view(num_candidates * batch_size, embedding_dim))
        candidate_encodings_before_mean = torch.cat([candidate_title_encodings, candidate_body_encodings])
        # candidate_encodings = torch.mean(candidate_encodings_before_mean, dim=0)

        q_title_recon = self.reconstructor(q_title_tensors)
        q_body_recon = self.reconstructor(q_body_tensors)
        q_recon_before_mean = torch.cat([q_title_recon, q_body_recon])
        # q_recon = torch.mean(q_recon_before_mean, dim=0)

        # get the encodings for the flattened out candidate tensors
        candidate_title_recons = self.reconstructor(candidate_title_tensors.view(num_candidates * batch_size, embedding_dim))
        candidate_body_recons = self.reconstructor(candidate_body_tensors.view(num_candidates * batch_size, embedding_dim))
        candidate_encodings_before_mean = torch.cat([candidate_title_encodings, candidate_body_encodings])
        # candidate_encodings = torch.mean(candidate_encodings_before_mean, dim=0)

        unpooled = torch.cat([q_encoding_before_mean, candidate_encodings_before_mean])
        recon = torch.cat([q_recon_before_mean, candidate_encodings_before_mean])
        return unpooled, recon
