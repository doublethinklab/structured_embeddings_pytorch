import unittest

import numpy as np

from sdemb import data
from sdemb.models import bern_struc
from tests import fake_data
import torch


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.raw_data = fake_data.FakeData()
        data.Corpus.create_folders(self.raw_data)
        np.random.seed(42)
        self.corpus = data.Corpus.from_data(self.raw_data)

    def test_get_batch(self):
        dataset = bern_struc.Dataset(self.corpus, 'dev', 2)
        batch1 = dataset[0]
        batch2 = dataset[1]
        expected1 = [
            torch.LongTensor([0, 1, 2, 3, 4]),
            torch.LongTensor([2, 3, 4, 5]),
        ]
        expected2 = [
            torch.LongTensor([5, 6, 7, 8]),
            torch.LongTensor([6, 7, 8]),
        ]
        for i in range(2):
            self.assertTrue(torch.allclose(expected1[i], batch1[i]))
            self.assertTrue(torch.allclose(expected2[i], batch2[i]))

    def tearDown(self):
        self.raw_data.remove_test_data()


class TestCollate(unittest.TestCase):

    def setUp(self):
        self.raw_data = fake_data.FakeData(
            min_tok_count=1,
            n_vocab=10)
        data.Corpus.create_folders(self.raw_data)
        np.random.seed(42)
        torch.random.manual_seed(42)

    def test_collate_batch(self):
        corpus = data.Corpus.from_data(self.raw_data)
        dataset = bern_struc.Dataset(corpus, 'dev', 2)
        collate = bern_struc.Collate(
            n_context=2, token_probs=corpus.vocab.probs(), n_negative_samples=2)
        batch_data = dataset[0]
        # [torch.LongTensor([0, 1, 2, 3, 4]),
        #  torch.LongTensor([2, 3, 4, 5])]
        g1, g2 = collate(batch_data)
        self.assertTrue(
            torch.allclose(
                torch.LongTensor([1, 2, 3]),
                g1.target_ixs))
        self.assertTrue(
            torch.allclose(
                torch.LongTensor([[0, 2], [1, 3], [2, 4]]),
                g1.context_ixs))
        self.assertTrue(
            torch.allclose(
                torch.LongTensor([[0, 1], [2, 0], [5, 4]]),
                g1.negative_sample_ixs))
        self.assertTrue(
            torch.allclose(
                torch.LongTensor([3, 4]),
                g2.target_ixs))
        self.assertTrue(
            torch.allclose(
                torch.LongTensor([[2, 4], [3, 5]]),
                g2.context_ixs))
        self.assertTrue(
            torch.allclose(
                torch.LongTensor([[8, 7], [1, 2]]),
                g2.negative_sample_ixs))

    def test_get_sample_probs(self):
        logits = np.array([0.25, 0.75])
        sample_probs = bern_struc.Collate.get_sample_probs(logits)
        expected = torch.Tensor([0.30492388, 0.69507612])
        self.assertTrue(torch.allclose(expected, sample_probs))

    def test_get_target_ixs(self):
        corpus = data.Corpus.from_data(self.raw_data)
        collate = bern_struc.Collate(2, corpus.vocab.probs(), 2)
        token_ixs = torch.Tensor([1, 3, 4, 1])
        target_ixs = collate.get_target_ixs(token_ixs)
        expected = torch.Tensor([3, 4])
        self.assertTrue(torch.allclose(expected, target_ixs))

    def test_get_context_ixs(self):
        corpus = data.Corpus.from_data(self.raw_data)
        collate = bern_struc.Collate(2, corpus.vocab.probs(), 2)
        token_ixs = torch.Tensor([1, 3, 4, 1])
        context_ixs = collate.get_context_ixs(token_ixs)
        expected = torch.Tensor([
            [1., 4.],
            [3., 1.]
        ])
        self.assertTrue(torch.allclose(expected, context_ixs))

    def test_get_negative_sample_ixs(self):
        corpus = data.Corpus.from_data(self.raw_data)
        collate = bern_struc.Collate(2, corpus.vocab.probs(), 2)
        token_ixs = torch.Tensor([1, 3, 4, 1])
        negative_sample_ixs = collate.get_negative_sample_ixs(token_ixs)
        expected = torch.LongTensor([[0, 1], [2, 0]])
        self.assertTrue(torch.allclose(expected, negative_sample_ixs))

    def tearDown(self):
        self.raw_data.remove_test_data()


class TestModel(unittest.TestCase):

    def setUp(self):
        self.raw_data = fake_data.FakeData()
        self.corpus = data.Corpus.from_data(self.raw_data)
        self.corpus.create_folders(self.raw_data)
        self.dataset = bern_struc.Dataset(self.corpus, 'dev', 2)
        self.collate = bern_struc.Collate(2, self.corpus.vocab.probs(), 2)
        torch.random.manual_seed(42)

    def test_init(self):
        _ = bern_struc.HierarchicalBernoulliEmbeddings(
            n_context=2, n_negative_samples=2, n_vocab=len(self.corpus.vocab),
            n_dim=3, n_groups=self.corpus.n_groups, sigma=0.1)
        self.assertTrue(True)  # check for a crash only

    def test_forward(self):
        model = bern_struc.HierarchicalBernoulliEmbeddings(
            n_context=2, n_negative_samples=2, n_vocab=len(self.corpus.vocab),
            n_dim=3, n_groups=self.corpus.n_groups, sigma=0.1)
        batch = self.collate(self.dataset[0])
        loss = model(batch)
        self.assertTrue(True)  # just check for exception

    def test_loss_with_group_embeds(self):
        model = bern_struc.HierarchicalBernoulliEmbeddings(
            n_context=2, n_negative_samples=2, n_vocab=len(self.corpus.vocab),
            n_dim=3, n_groups=self.corpus.n_groups, sigma=0.1)
        model.init_group_embeddings()
        batch = self.collate(self.dataset[0])
        loss = model(batch)
        self.assertTrue(True)  # just check for exception

    def test_fit(self):
        model = bern_struc.HierarchicalBernoulliEmbeddings(
            n_context=2, n_negative_samples=2, n_vocab=len(self.corpus.vocab),
            n_dim=3, n_groups=self.corpus.n_groups, sigma=0.1)
        model.fit(self.corpus, 0.01, 2, 5)

    def tearDown(self):
        self.raw_data.remove_test_data()
