import unittest

from sdemb import data
from sdemb.models import bern_struc
from tests import fake_data


class TestDataset(unittest.TestCase):

    def setUp(self):
        raw_data = fake_data.FakeData()
        data.Corpus.create_folders(raw_data)

    def test_get_batch(self):
        corpus = data.Corpus.from_data(fake_data.FakeData())
        dataset = bern_struc.Dataset(corpus, 'train', 2)
        batch1 = dataset[0]
        batch2 = dataset[1]
        print(batch1)
        self.assertEqual(2, len(batch1))  # n groups
        self.assertEqual(2, len(batch1[0]))
        self.assertEqual(2, len(batch1[2]))
        self.assertEqual(2, len(batch2))
        self.assertEqual(2, len(batch2[0]))
        self.assertEqual(2, len(batch2[2]))

    def tearDown(self):
        fake_data.FakeData.remove_test_data()
