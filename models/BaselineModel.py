from sklearn.metrics.pairwise import cosine_similarity

class BaselineModel():

    def compute(self,
                q_tensors,
                candidate_tensors):

        num_candidates, batch_size, embedding_dim = candidate_tensors.shape

        flattened_candidate_tensors = candidate_tensors.view(num_candidates*batch_size,embedding_dim)

        # expand and flatten q encodings so that we can pass it through cosine similarities with flattened out
        # candidate encodings
        expanded_q_tensor = q_tensors.view(1,batch_size,embedding_dim)
        expanded_q_tensor = expanded_q_tensor.expand(num_candidates,batch_size,embedding_dim)
        expanded_q_tensor = expanded_q_tensor.contiguous().view(num_candidates*batch_size,embedding_dim)

        # compute cosine similarities
        expanded_cosine_similarities = cosine_similarity(expanded_q_tensor, flattened_candidate_tensors)

        # un-flatten cosine similarities so that we get output of size batch_size * num_candidates (t() is transpose)
        output = expanded_cosine_similarities.view(num_candidates,batch_size,1).view(num_candidates,batch_size).t()

        return output