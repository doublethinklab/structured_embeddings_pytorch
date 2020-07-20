import shutil
import string

from sdemb import data


class FakeData(data.RawData):

    def __init__(self):
        super().__init__(
            corpus_name='test',
            group_names=['g1', 'g2'],
            min_tok_count=2,
            n_vocab=2,
            subsample_threshold=0.5)

    def doc_ids(self, group_name):
        return [f'{group_name}.{i}' for i in range(1, 4)]

    def doc_tokens(self, doc_id):
        # all groups end up with three docs:
        # [c, d]
        # [e, f, g, h]
        # [g, h, i, j, k, l]
        # so counts are {g: 2, h: 2}, making it to the vocab.
        i = int(doc_id.split('.')[-1])
        n = i * 2
        return [string.ascii_lowercase[j+n] for j in range(n)]

    @staticmethod
    def remove_test_data():
        shutil.rmtree('data/test')
