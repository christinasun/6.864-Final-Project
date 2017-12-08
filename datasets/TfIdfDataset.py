import sys
from os.path import dirname, realpath
import torch
import torch.utils.data as data
sys.path.append(dirname(dirname(realpath(__file__))))
import utils.ubuntu_data_utils as ubuntu_data_utils
import utils.android_data_utils as android_data_utils
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfDataset(data.Dataset):

    def __init__(self, name, source):
        self.vectorizer = TfidfVectorizer()
        self.name = name
        self.dataset = []
        self.source = source
        if self.source == 'ubuntu':
            self.get_dev_examples = ubuntu_data_utils.get_dev_examples
            self.get_test_examples = ubuntu_data_utils.get_test_examples
            self.data_dict = ubuntu_data_utils.get_data_dict()
        if self.source == 'android':
            self.get_dev_examples = android_data_utils.get_dev_examples
            self.get_test_examples = android_data_utils.get_test_examples
            self.data_dict = android_data_utils.get_data_dict()

        corpus = []
        for title, body in self.data_dict.values():
            title_and_body = ' '.join([title, body])
            corpus.append(title_and_body)
        self.vectorizer.fit_transform(corpus)


        # TODO: decide if we need to implement this for train data (I don't think we do)
        if name == 'dev':
            dev_examples = self.get_dev_examples()
            for example in dev_examples:
                self.update_dataset_from_dev_or_test_example(example)
        elif name == 'test':
            test_examples = self.get_test_examples()
            for example in test_examples:
                self.update_dataset_from_dev_or_test_example(example)

    def update_dataset_from_dev_or_test_example(self, example):
        if self.source == 'ubuntu':
            qid, similar_qids, candidate_qids, BM25_scores = example
        elif self.source == 'android':
            qid, similar_qids, candidate_qids = example

        qid_tfidf_tensor = self.get_tfidf_tensor(' '.join(self.data_dict[qid]))
        candidate_tfidf_tensors = [self.get_tfidf_tensor(' '.join(self.data_dict[cqid])) for cqid in candidate_qids]

        if self.source == 'ubuntu':
            labels = [1 if cqid in similar_qids else 0 for cqid in candidate_qids]
            sample = \
                {'qid': qid,
                 'similar_qids': similar_qids,
                 'candidates': candidates,
                 'qid_tfidf_tensor': qid_tfidf_tensor,
                 'candidate_tfidf_tensors': candidate_tfidf_tensors,
                 'labels': labels
                 }
            self.dataset.append(sample)

        elif self.source == 'android':
            for similar_qid in similar_qids:
                similar_tfidf_tensor = self.get_tfidf_tensor(' '.join(self.data_dict[similar_qid]))
                candidates = [similar_qid] + candidate_qids
                labels = [1 if cqid in similar_qids else 0 for cqid in candidates]
                sample = \
                    {'qid': qid,
                     'similar_qids': similar_qids,
                     'candidates': candidates,
                     'qid_tfidf_tensor': qid_tfidf_tensor,
                     'candidate_tfidf_tensors': [similar_tfidf_tensor] + candidate_tfidf_tensors,
                     'labels': labels
                     }
                self.dataset.append(sample)
        return

    def update_dataset_from_train_example(self, example):
        # adds samples to dataset for each training example
        # each training example generates multiple samples
        qid, similar_qids, random_qids = example
        qid_tensors = map(self.get_indices_tensor, self.data_dict[qid])

        random_candidate_tensors = [map(self.get_indices_tensor, self.data_dict[cqid]) for cqid in random_qids]
        random_candidate_title_tensors, random_candidate_body_tensors = zip(*random_candidate_tensors)
        random_candidate_title_tensors = list(random_candidate_title_tensors)
        random_candidate_body_tensors = list(random_candidate_body_tensors)

        for similar_qid in similar_qids:
            candidates = [similar_qid] + random_qids
            similar_qid_tensors = map(self.get_indices_tensor, self.data_dict[similar_qid])

            sample = {'qid': qid,
                      'candidates': candidates,
                      'qid_title_tensor': qid_tensors[0],
                      'qid_body_tensor': qid_tensors[1],
                      'candidate_title_tensors': [similar_qid_tensors[0]] + random_candidate_title_tensors,
                      'candidate_body_tensors': [similar_qid_tensors[1]] + random_candidate_body_tensors
                      }
            self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample

    def get_tfidf_tensor(self, text_arr):
        o = self.vectorizer.transform([text_arr])
        # print "new"
        # print o
        return o