import shutil
import string

from sdemb import data


class FakeData(data.RawData):

    def __init__(self, min_tok_count=1, n_vocab=24):
        super().__init__(
            corpus_name='test',
            group_names=['g1', 'g2'],
            min_tok_count=min_tok_count,
            n_vocab=n_vocab,
            subsample_threshold=0.5)

    def doc_ids(self, group_name):
        return [f'{group_name}.{i}' for i in range(1, 4)]

    def doc_tokens(self, doc_id):
        # g1:
        # g1.1: [b c d e f g h i j k]
        # g1.2: [c d e f g h i j k l]
        # g1.3: [d e f g h i j k l m n o]
        # g2:
        # g2.1: [c d e f g h i j k l]
        # g2.2: [d e f g h i j k l m n o p q r s]
        # g2.3: [e g h i j k l m n o p q r s t u v w x]
        g = int(doc_id[1])
        i = int(doc_id.split('.')[-1])
        a = g * i
        b = min(11, max(g * i * 4, 21))
        return [string.ascii_lowercase[j] for j in range(a, b)]

    @staticmethod
    def remove_test_data():
        shutil.rmtree('data/test')
