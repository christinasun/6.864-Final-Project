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
        self.embedding_layer.requires_grad = False

        self.hidden_dim = args.hidden_dim
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

        expanded_q_embedding = q_embedding.view(1,d2,self.hidden_dim).expand(num_candidates,d2,self.hidden_dim)\
            .contiguous().view(num_candidates*d2,self.hidden_dim)

        expanded_cosine_similarites = F.cosine_similarity(expanded_q_embedding, candidate_embeddings, dim=1)

        # TODO: Double check that this transformation is resizing the output as desired
        output = expanded_cosine_similarites.view(num_candidates,d2,1).view(num_candidates,d2).t()


        return output

    def forward_helper(self, tensor):
        pass


class CNN(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args):
        super(CNN, self).__init__(embeddings, args)
        vocab_size, embed_dim = embeddings.shape
        self.conv = nn.Conv2d(1, self.hidden_dim, (3, embed_dim), padding= (2,0))
        self.tanh = nn.Tanh()
        self.pooling = torch.nn.AvgPool2d((1,args.len_query))



    def forward_helper(self, tensor):
        x = self.embedding_layer(tensor) # (batch size, width (length of text), height (embedding dim))
        x = x.unsqueeze(1) # (batch size, 1, width, height)

        hiddens = self.conv(x).squeeze(3)
        tanh_x = self.tanh(hiddens)
        pooled = self.pooling(tanh_x)
        output = pooled.squeeze(2)
        return output
        

class LSTM(AbstractAskUbuntuModel):

    def __init__(self, embeddings, args):
        super(LSTM, self).__init__(embeddings, args)

    def forward(self):
        # TODO: Implement
        pass
