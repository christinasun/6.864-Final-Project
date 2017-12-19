import torch
import torch.nn as nn


class Adversary(nn.Module):
    # The adversary abstracts away the encoder and the domain classifier.
    # It takes in pairs of title/body tensors, encodes them using the encoder, and then feeds the encodings into the
    # domain classifier. The adversary outputs the predicted labels for each title/body pair.

    def __init__(self, args, encoder, domain_classifier):
        super(Adversary, self).__init__()

        self.args = args
        self.encoder = encoder
        self.domain_classifier = domain_classifier
        self.hidden_dim = args.hidden_dim
        return

    def forward(self, for_dc_title_tensors, for_dc_body_tensors):
        # get the encodings for the flattened out domain classifier tensors
        num_for_dc, batch_size, embedding_dim = for_dc_title_tensors.size()
        for_dc_title_encodings = self.encoder(for_dc_title_tensors.view(num_for_dc * batch_size, embedding_dim))
        for_dc_body_encodings = self.encoder(for_dc_title_tensors.view(num_for_dc * batch_size, embedding_dim))
        for_dc_encodings_before_mean = torch.stack([for_dc_title_encodings, for_dc_body_encodings])
        for_dc_encodings = torch.mean(for_dc_encodings_before_mean, dim=0)

        expanded_labels = self.domain_classifier(for_dc_encodings)

        dc_output = expanded_labels.view(num_for_dc, batch_size, 1).view(num_for_dc, batch_size).t()
        return dc_output
