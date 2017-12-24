import torch
import torch.nn.functional as F
import torch.nn as nn

class Reconstructor(nn.Module):
    def __init__(self, encoder):
        super(Reconstructor, self).__init__()
        self.args = encoder.args
        self.encoder = encoder
        self.hidden_dim = self.args.hidden_dim
        return

    def forward(self,
                q_title_tensors,
                q_body_tensors,
                candidate_title_tensors,
                candidate_body_tensors):
        q_title_encodings, qte_losses = self.encoder(q_title_tensors)
        q_body_encodings, qbe_losses = self.encoder(q_body_tensors)
        q_losses = torch.cat([qte_losses, qbe_losses])

        # get the encodings for the flattened out candidate tensors
        num_candidates, batch_size, embedding_dim = candidate_title_tensors.size()
        candidate_title_encodings, cte_losses = self.encoder(candidate_title_tensors.view(num_candidates * batch_size, embedding_dim))
        candidate_body_encodings, cbe_losses = self.encoder(candidate_body_tensors.view(num_candidates * batch_size, embedding_dim))
        candidate_losses = torch.cat([cte_losses, cbe_losses])

        losses = torch.cat([q_losses, candidate_losses])
        return losses

        # q_title_encodings, qte_x_hat, qte_x_perm = self.encoder(q_title_tensors)
        # q_body_encodings, qbe_x_hat, qbe_x_perm = self.encoder(q_body_tensors)
        # q_x_hats = torch.cat([qte_x_hat, qbe_x_hat])
        # q_x_perms = torch.cat([qte_x_perm, qbe_x_perm])

        # # get the encodings for the flattened out candidate tensors
        # num_candidates, batch_size, embedding_dim = candidate_title_tensors.size()
        # candidate_title_encodings, cte_x_hat, cte_x_perm  = self.encoder(candidate_title_tensors.view(num_candidates * batch_size, embedding_dim))
        # candidate_body_encodings, cbe_x_hat, cbe_x_perm  = self.encoder(candidate_body_tensors.view(num_candidates * batch_size, embedding_dim))
        # c_x_hats = torch.cat([cte_x_hat, cbe_x_hat])
        # c_x_perms = torch.cat([cte_x_perm, cbe_x_perm])

        # return torch.cat([q_x_hats, c_x_hats]), torch.cat([q_x_perms, c_x_perms])
