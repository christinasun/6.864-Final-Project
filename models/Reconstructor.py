import torch
import torch.nn as nn


class Reconstructor(nn.Module):

    def __init__(self, encoder):
        super(Reconstructor, self).__init__()

        self.args = encoder.args
        self.encoder = encoder
        self.hidden_dim = self.args.hidden_dim
        return

    def forward(self, for_dc_title_tensors, for_dc_body_tensors):
        # get the encodings for the flattened out domain classifier tensors
        num_for_dc, batch_size, embedding_dim = for_dc_title_tensors.size()
        for_dc_title_encodings = self.encoder(for_dc_title_tensors.view(num_for_dc * batch_size, embedding_dim))
        for_dc_body_encodings = self.encoder(for_dc_title_tensors.view(num_for_dc * batch_size, embedding_dim))
        for_dc_encodings_before_mean = torch.stack([for_dc_title_encodings, for_dc_body_encodings])
        for_dc_encodings = torch.mean(for_dc_encodings_before_mean, dim=0)

        return for_dc_encodings
