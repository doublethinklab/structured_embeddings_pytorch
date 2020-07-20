import json
import os
import shutil
import unittest

import numpy as np

from sdemb import data
from tests import fake_data


test_groups = ['g1', 'g2']
test_docs = [
    # g1
    ['a', 'b', 'c', 'd', 'c'],
    ['a', 'b', 'a', 'a', 'a'],
    ['c', 'b', 'c', 'c', 'c'],
    # g2
    ['a', 'b', 'c', 'd'],
    ['a', 'a', 'a', 'b'],
    ['c', 'c', 'c', 'b'],
]
test_tok_ixs = [
    [1, 2, 3, 4, 3],
    [1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3],
    [1, 3, 3, 1],
    [1, 1, 1, 1],
    [3, 3, 3, 3],
]


def create_folders():
    corpus_dir = 'data/test'
    if not os.path.exists(corpus_dir):
        os.mkdir(corpus_dir)
    for group_name in test_groups:
        group_dir = os.path.join(corpus_dir, group_name)
        if not os.path.exists(group_dir):
            os.mkdir(group_dir)


def remove_folders():
    corpus_dir = 'data/test'
    if os.path.exists(corpus_dir):
        shutil.rmtree(corpus_dir)


class TestIxDict(unittest.TestCase):

    def test_entities_come_out_in_ix_order(self):
        items = ['b', 'c', 'a']
        ix_dict = data.IxDict(items)
        entities = ix_dict.entities
        ixs = [ix_dict[x] for x in entities]
        expected = [0, 1, 2]
        self.assertEqual(ixs, expected)


class TestVocab(unittest.TestCase):

    def setUp(self):
        create_folders()

    def test_doc2bow(self):
        counts = {'a': 4, 'b': 3}
        vocab = data.Vocab('test', counts)
        bow = vocab.doc2ixs(['a', 'c', 'b'])
        expected = [0, 1]
        self.assertEqual(expected, bow)

    def test_save(self):
        counts = {'a': 4, 'b': 3}
        vocab = data.Vocab('test', counts)
        vocab.save()
        with open(data.Vocab.file_path('test')) as f:
            jdata = json.loads(f.read())
        self.assertEqual('test', jdata['corpus_name'])
        self.assertEqual(counts, jdata['counts'])

    def test_load(self):
        counts = {'a': 4, 'b': 3}
        vocab = data.Vocab('test', counts)
        vocab.save()
        vocab = data.Vocab.load('test')
        self.assertEqual('test', vocab.corpus_name)
        self.assertEqual(counts, vocab.counts)

    def test_logits(self):
        counts = {'a': 1, 'b': 2, 'c': 1}
        vocab = data.Vocab('test', counts)
        expected = np.log(np.array([0.25, 0.5, 0.25]))
        logits = vocab.logits()
        self.assertTrue(np.array_equal(expected, logits))

    def test_probs(self):
        counts = {'a': 1, 'b': 2, 'c': 1}
        vocab = data.Vocab('test', counts)
        expected = np.array([0.25, 0.5, 0.25])
        probs = vocab.probs()
        self.assertTrue(np.array_equal(expected, probs))

    def tearDown(self):
        remove_folders()


class TestDoc(unittest.TestCase):

    def setUp(self):
        create_folders()

    def test_save(self):
        doc = data.Doc('test', 'g1', 'g1.1', test_tok_ixs[0])
        doc.save()
        file_path = doc.file_path('test', 'g1', 'g1.1')
        with open(file_path) as f:
            jdata = json.loads(f.read())
        self.assertEqual('test', jdata['corpus_name'])
        self.assertEqual('g1', jdata['group_name'])
        self.assertEqual('g1.1', jdata['doc_id'])
        self.assertEqual([1, 2, 3, 4, 3], jdata['token_ixs'])

    def test_load(self):
        doc = data.Doc('test', 'g1', 'g1.1', test_tok_ixs[0])
        doc.save()
        doc = data.Doc.load('test', 'g1', 'g1.1')
        self.assertEqual('test', doc.corpus_name)
        self.assertEqual('g1', doc.group_name)
        self.assertEqual('g1.1', doc.doc_id)
        self.assertEqual([1, 2, 3, 4, 3], doc.token_ixs)

    def test_iter(self):
        doc = data.Doc('test', 'g1', 'g1.1', test_tok_ixs[0])
        mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 3}
        for ix, tok_ix in enumerate(doc):
            self.assertEqual(mapping[ix], tok_ix)

    def tearDown(self):
        remove_folders()


class TestGroup(unittest.TestCase):

    def setUp(self):
        create_folders()

    def get_docs(self):
        docs = []
        doc_ids = ['g1.1', 'g1.2', 'g1.3']
        for doc_id, doc_toks in zip(doc_ids, test_tok_ixs):
            doc = data.Doc('test', 'g1', doc_id, doc_toks)
            doc.save()
            docs.append(doc)
        return docs

    def test_save(self):
        group = data.Group.from_docs('test', 'g1', self.get_docs())
        group.save()
        with open(data.Group.file_path('test', 'g1')) as f:
            jdata = json.loads(f.read())
        self.assertEqual('test', jdata['corpus_name'])
        self.assertEqual('g1', jdata['name'])
        self.assertEqual(['g1.1', 'g1.2', 'g1.3'], jdata['doc_ids'])
        self.assertEqual(15, jdata['n_tokens'])

    def test_load(self):
        group = data.Group.from_docs('test', 'g1', self.get_docs())
        group.save()
        group = data.Group.load('test', 'g1')
        self.assertEqual('test', group.corpus_name)
        self.assertEqual('g1', group.name)
        doc_ids = {
            'train': ['g1.1'],
            'dev': ['g1.2'],
            'test': ['g1.3'],
        }
        self.assertEqual(doc_ids, group.doc_ids)
        self.assertEqual(15, group.n_tokens)

    def test_get_splits(self):
        doc_ids = data.Group.get_splits(list(range(6)))
        expected = {
            'train': [0, 1],
            'dev': [2, 3],
            'test': [4, 5],
        }
        self.assertEqual(expected, doc_ids)

    def test_get_batch(self):
        group = data.Group.from_docs('test', 'g1', self.get_docs())
        expected = [
            [1, 2, 3],
            [4, 3]
        ]
        for ix in range(2):
            batch = group.get_batch('train', 3)
            self.assertEqual(expected[ix], batch)
        # test it resets
        for ix in range(2):
            batch = group.get_batch('train', 3)
            self.assertEqual(expected[ix], batch)

    def tearDown(self):
        remove_folders()


class TestCorpus(unittest.TestCase):

    def setUp(self):
        create_folders()

    def test_from_data(self):
        raw_data = fake_data.FakeData()
        corpus = data.Corpus.from_data(raw_data)
        # TODO: this also mostly tests absence of exceptions
        self.assertEqual(2, len(corpus.groups))

    def test_create_vocab(self):
        raw_data = fake_data.FakeData()
        vocab = data.Corpus.create_vocab(raw_data)
        expected = {'g': 2, 'h': 2}
        self.assertEqual(expected, vocab.counts)

    def test_create_group_dict(self):
        raw_data = fake_data.FakeData()
        data.Corpus.create_group_dict(raw_data)
        group_dict = data.GroupDict.load(raw_data.corpus_name)
        expected = ['g1', 'g2']
        self.assertEqual(expected, group_dict.entities)

    def test_create_doc_dict(self):
        raw_data = fake_data.FakeData()
        data.Corpus.create_doc_dict(raw_data)
        doc_dict = data.DocDict.load(raw_data.corpus_name)
        expected = ['g1.1', 'g1.2', 'g1.3',
                    'g2.1', 'g2.2', 'g2.3']
        self.assertEqual(expected, doc_dict.entities)

    def test_save(self):
        raw_data = fake_data.FakeData()
        corpus = data.Corpus.from_data(raw_data)
        corpus.save()
        with open(corpus.file_path(corpus.name)) as f:
            jdata = json.loads(f.read())
        self.assertEqual('test', jdata['name'])
        self.assertEqual(2, jdata['min_tok_count'])
        self.assertEqual(2, jdata['n_vocab'])
        self.assertEqual(0.5, jdata['subsample_threshold'])

    def test_load(self):
        raw_data = fake_data.FakeData()
        corpus = data.Corpus.from_data(raw_data)
        corpus.save()
        corpus = data.Corpus.load('test')
        self.assertEqual('test', corpus.name)
        self.assertEqual(2, corpus.min_tok_count)
        self.assertEqual(2, corpus.n_vocab)
        self.assertEqual(0.5, corpus.subsample_threshold)

    def test_load_groups(self):
        raw_data = fake_data.FakeData()
        corpus = data.Corpus.from_data(raw_data)
        groups = corpus.load_groups()
        self.assertEqual(2, len(groups))
        expected =['g1', 'g2']
        self.assertEqual(expected, [x.name for x in groups])

    def test_parse_groups_and_docs(self):
        np.random.seed(42)
        raw_data = fake_data.FakeData()
        vocab = data.Corpus.create_vocab(raw_data)
        data.Corpus.parse_groups_and_docs(raw_data, vocab)
        # TODO: this really just tests it ran without error, details skipped
        g1 = data.Group.load('test', 'g1')
        g2 = data.Group.load('test', 'g2')
        self.assertEqual(3, g1.n_docs)
        self.assertEqual(3, g2.n_docs)
        self.assertEqual(4, g1.n_tokens)
        self.assertEqual(4, g2.n_tokens)

    def test_subsample(self):
        token_ixs = [0, 0, 0, 1, 0]
        token_logits = np.array([0.99, 1e-10])
        np.random.seed(42)
        token_ixs = data.Corpus.subsample(token_ixs, token_logits, 10**(-5))
        expected = [1]
        self.assertEqual(expected, token_ixs)

    def tearDown(self):
        remove_folders()
