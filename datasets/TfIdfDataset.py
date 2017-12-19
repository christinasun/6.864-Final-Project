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
            qid, positive_qids, candidate_qids, BM25_scores = example
            negative_qids = [cpid for cpid in candidate_qids if cpid not in similar_qids]

        elif self.source == 'android':
            qid, positive_qids, negative_qids = example

        qid_tfidf_tensor = self.get_tfidf_tensor(' '.join(self.data_dict[qid]))
        positive_tfidf_tensors = [self.get_tfidf_tensor(' '.join(self.data_dict[cqid])) for cqid in positive_qids]
        negative_tfidf_tensors = [self.get_tfidf_tensor(' '.join(self.data_dict[cqid])) for cqid in negative_qids]

        sample = \
            {'qid': qid,
             'positive_qids': positive_qids,
             'negative_qids': negative_qids,
             'qid_tfidf_tensor': qid_tfidf_tensor,
             'negative_tfidf_tensors': negative_tfidf_tensors,
             'positive_tfidf_tensors': positive_tfidf_tensors
             }
        self.dataset.append(sample)
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index]

    def get_tfidf_tensor(self, text_arr):
        return self.vectorizer.transform([text_arr])