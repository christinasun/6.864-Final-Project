import torch
import torch.nn.functional as F
import torch.nn as nn


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.model_name == 'cnn':
        return CNN(embeddings, args)
    elif args.model_name == 'lstm':
        return LSTM(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


class AbstractAskUbuntuModel(nn.Module):

    def __init__(self, embeddings, args):
        super(AbstractAskUbuntuModel, self).__init__()

        self.args = args
        vocab_size, embed_dim = embeddings.shape

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.W_hidden = nn.Linear(embed_dim, 200)
        self.W_out = nn.Linear(200, 200)
        return

    def forward(self, q_title_tensors, q_body_tensors, candidate_title_tensors, candidate_body_tensors):
        q_title_embeddings = self.forward_helper(q_title_tensors)
        q_body_embeddings = self.forward_helper(q_body_tensors)

        q_embedding_before_mean = torch.stack([q_title_embeddings, q_body_embeddings])

        q_embedding = torch.mean(q_embedding_before_mean, dim=0)

        num_candidates, d2, d3 = candidate_title_tensors.size()

        title_embeddings = self.forward_helper(candidate_title_tensors.view(num_candidates*d2,d3))
        body_embeddings = self.forward_helper(candidate_body_tensors.view(num_candidates*d2,d3))
        candidate_embeddings_before_mean = torch.stack([title_embeddings, body_embeddings])
        candidate_embeddings = torch.mean(candidate_embeddings_before_mean, dim=0)

        expanded_q_embedding = q_embedding.view(1,d2,d3).expand(num_candidates,d2,d3).contiguous().view(num_candidates*d2,d3)

        expanded_cosine_similarites = F.cosine_similarity(expanded_q_embedding, candidate_embeddings, dim=1)

        # TODO: Double check that this transformation is resizing the output as desired
        output = expanded_cosine_similarites.view(num_candidates,d2,1).view(num_candidates,d2).t()


        return output

    def forward_helper(self, tensor):
        pass


class CNN(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args):
        super(CNN, self).__init__(embeddings, args)

    def forward_helper(self, tensor):
        # TODO: Implement the CNN
        all_x = self.embedding_layer(tensor)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu( self.W_hidden(avg_x) )
        out = self.W_out(hidden)
        return out
        


class LSTM(AbstractAskUbuntuModel):


    def __init__(self, embeddings, args):
        super(LSTM, self).__init__(embeddings, args)

    def forward(self):
        # TODO: Implement
        pass
