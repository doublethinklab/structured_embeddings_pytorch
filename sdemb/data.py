"""Data preprocessing functions and data structures.

Corpus 1---*> Group 1---*> Doc.
"""
from collections import Counter
import json
import math
import os
import types
from typing import Mapping, Sequence, Union

import numpy as np
from tqdm.notebook import tqdm


data_dir = 'data'


class IxDict:

    def __init__(self, items: Sequence[str]):
        items = list(sorted(items))
        self.item_to_ix = dict(zip(items, range(len(items))))
        self.ix_to_item = dict(zip(range(len(items)), items))

    def __getitem__(self, item: Union[str, int]):
        if isinstance(item, str):
            if item not in self.item_to_ix:
                raise ValueError(f'Not in vocab: {item}')
            return self.item_to_ix[item]
        if isinstance(item, int):
            if item not in self.ix_to_item:
                raise ValueError(f'Not in vocab: {item}')
            return self.ix_to_item[item]
        raise ValueError(f'Unexpected item type: {type(item)}')

    def __len__(self):
        return len(self.ix_to_item)

    @property
    def entities(self) -> Sequence[str]:
        # ensures goes out in ix order
        return list(sorted(self.item_to_ix.keys()))


class GroupDict(IxDict):

    def __init__(self, group_names: Sequence[str], corpus_name: str):
        super().__init__(list(sorted(group_names)))
        self.corpus_name = corpus_name

    @staticmethod
    def file_path(corpus_name):
        return f'data/{corpus_name}/group.dict'

    @classmethod
    def load(cls, corpus_name: str):
        with open(cls.file_path(corpus_name)) as f:
            data = json.loads(f.read())
            return cls(**data)

    def save(self):
        with open(self.file_path(self.corpus_name), 'w+') as f:
            data = {
                'group_names': list(sorted(self.item_to_ix.keys())),
                'corpus_name': self.corpus_name,
            }
            f.write(json.dumps(data))


class DocDict(IxDict):

    def __init__(self, doc_ids: Sequence[str], corpus_name: str):
        super().__init__(doc_ids)
        self.corpus_name = corpus_name

    @staticmethod
    def file_path(corpus_name):
        return f'data/{corpus_name}/doc.dict'

    @classmethod
    def load(cls, corpus_name: str):
        with open(cls.file_path(corpus_name)) as f:
            data = json.loads(f.read())
            return cls(**data)

    def save(self):
        with open(self.file_path(self.corpus_name), 'w+') as f:
            data = {
                'doc_ids': list(sorted(self.item_to_ix.keys())),
                'corpus_name': self.corpus_name,
            }
            f.write(json.dumps(data))


class Vocab(IxDict):

    def __init__(self, corpus_name: str, counts: Mapping[str, int]):
        super().__init__(list(counts.keys()))
        self.corpus_name = corpus_name
        self.counts = counts
        self.n = sum(counts.values())

    def doc2ixs(self, tokens):
        ixs = []
        for token in tokens:
            if token in self.item_to_ix:
                ixs.append(self[token])
        return ixs

    @staticmethod
    def file_path(corpus_name):
        return f'{data_dir}/{corpus_name}/vocab.json'

    @classmethod
    def load(cls, corpus_name):
        file_path = cls.file_path(corpus_name)
        with open(file_path) as f:
            data = json.loads(f.read())
        return cls(**data)

    def logits(self):
        return np.log(self.probs())

    def probs(self):
        probs = np.zeros((len(self),))
        for tok_ix, tok in self.ix_to_item.items():
            probs[tok_ix] = self.counts[tok] / self.n
        return probs

    def save(self):
        file_path = self.file_path(self.corpus_name)
        with open(file_path, 'w+') as f:
            data = {
                'corpus_name': self.corpus_name,
                'counts': self.counts,
            }
            f.write(json.dumps(data))


class Doc:

    def __init__(self, corpus_name: str, group_name: str, doc_id: str,
                 token_ixs: Sequence[int]):
        self.corpus_name = corpus_name
        self.group_name = group_name
        self.doc_id = doc_id
        self.token_ixs = token_ixs
        self.n_tokens = len(token_ixs)

    def __iter__(self) -> int:
        for token_ix in self.token_ixs:
            yield token_ix

    def __len__(self):
        return self.n_tokens

    @staticmethod
    def file_path(corpus_name: str, group_name: str, doc_id: str) -> str:
        return f'{data_dir}/{corpus_name}/{group_name}/{doc_id}.json'

    @classmethod
    def load(cls, corpus_name: str, group_name: str, doc_id: str):
        with open(cls.file_path(corpus_name, group_name, doc_id)) as f:
            data = json.loads(f.read())
            return cls(**data)

    def save(self):
        file_path = self.file_path(
            self.corpus_name, self.group_name, self.doc_id)
        with open(file_path, 'w+') as f:
            data = {
                'corpus_name': self.corpus_name,
                'group_name': self.group_name,
                'doc_id': self.doc_id,
                'token_ixs': self.token_ixs,
            }
            f.write(json.dumps(data))


class Group:

    def __init__(self, corpus_name: str, name: str,
                 doc_ids: Mapping[str, Sequence[str]],
                 n_tokens: Mapping[str, int]):
        # NOTE: docs are already saved
        self.corpus_name = corpus_name
        self.name = name
        self.n_docs = sum(len(v) for v in doc_ids.values())
        self.n_tokens = n_tokens
        self.doc_ids = doc_ids
        self.generator = {
            'train': self.new_generator('train'),
            'dev': self.new_generator('dev'),
            'test': self.new_generator('test'),
        }

    def docs(self, subset) -> types.GeneratorType:
        for doc_id in self.doc_ids[subset]:
            yield Doc.load(self.corpus_name, self.name, doc_id)

    @staticmethod
    def file_path(corpus_name: str, name: str) -> str:
        return f'{data_dir}/{corpus_name}/{name}.json'

    @classmethod
    def from_docs(cls, corpus_name: str, name: str, docs: Sequence[Doc]):
        doc_ids = [x.doc_id for x in docs]
        doc_ids = cls.get_splits(doc_ids)
        n_tokens = {}
        for subset, ids in doc_ids.items():
            split_docs = [x for x in docs if x.doc_id in ids]
            n_tokens[subset] = sum(len(x) for x in split_docs)
        return cls(corpus_name, name, doc_ids, n_tokens)

    def get_batch(self, subset: str, batch_size: int) -> Sequence[int]:
        token_ixs = []
        end_reached = False
        while len(token_ixs) < batch_size and not end_reached:
            try:
                token_ixs.append(next(self.generator[subset]))
            except StopIteration:
                end_reached = True
                self.generator[subset] = self.new_generator(subset)
        return token_ixs

    @staticmethod
    def get_splits(doc_ids) -> Mapping[str, Sequence[str]]:
        m = max(int(len(doc_ids) / 3),
                math.floor(len(doc_ids) / 10))
        return {
            'train': doc_ids[0:-m * 2],
            'dev': doc_ids[-m * 2:-m],
            'test': doc_ids[-m:],
        }

    @classmethod
    def load(cls, corpus_name: str, name: str):
        file_path = cls.file_path(corpus_name, name)
        with open(file_path) as f:
            data = json.loads(f.read())
            return cls(**data)

    def load_doc_ids(self) -> Sequence[str]:
        directory = f'{data_dir}/{self.corpus_name}/{self.name}'
        return [x.split('.')[0] for x in os.listdir(directory)]

    def new_generator(self, subset) -> types.GeneratorType:
        for doc_id in self.doc_ids[subset]:
            doc = Doc.load(self.corpus_name, self.name, doc_id)
            for token in doc:
                yield token

    def save(self):
        file_path = self.file_path(self.corpus_name, self.name)
        with open(file_path, 'w+') as f:
            data = {
                'corpus_name': self.corpus_name,
                'name': self.name,
                'doc_ids': self.doc_ids,
                'n_tokens': self.n_tokens,
            }
            f.write(json.dumps(data))


class RawData:
    """Should be a iterator of iterators of iterators.

    Being: group -> document -> token.
    """

    def __init__(self, corpus_name: str, group_names: Sequence[str],
                 min_tok_count: int, n_vocab: int,
                 subsample_threshold: float = 10**(-5)):
        self.corpus_name = corpus_name
        self.group_names = group_names
        self.min_tok_count = min_tok_count
        self.n_vocab = n_vocab
        self.subsample_threshold = subsample_threshold

    def doc_ids(self, group_name):
        raise NotImplementedError

    def doc_tokens(self, doc_id):
        raise NotImplementedError

    def group_tokens(self, group_name):
        for doc_id in self.doc_ids(group_name):
            for token in self.doc_tokens(doc_id):
                yield token


class Corpus:

    def __init__(self, name: str, min_tok_count: int, n_vocab: int,
                 subsample_threshold: int = 10**(-5)):
        self.name = name
        self.min_tok_count = min_tok_count
        self.n_vocab = n_vocab
        self.subsample_threshold = subsample_threshold
        self.group_dict = GroupDict.load(name)
        self.n_groups = len(self.group_dict)
        self.doc_dict = DocDict.load(name)
        self.n_docs = len(self.doc_dict)
        self.vocab = Vocab.load(name)
        self.groups = self.load_groups()

    def __iter__(self):
        for group in self.groups:
            yield group

    @classmethod
    def from_data(cls, raw_data):
        cls.create_folders(raw_data)
        cls.create_group_dict(raw_data)
        cls.create_doc_dict(raw_data)
        vocab = cls.create_vocab(raw_data)
        cls.parse_groups_and_docs(raw_data, vocab)
        corpus = cls(name=raw_data.corpus_name,
                     min_tok_count=raw_data.min_tok_count,
                     n_vocab=raw_data.n_vocab,
                     subsample_threshold=raw_data.subsample_threshold)
        corpus.save()
        return corpus

    @staticmethod
    def create_folders(raw_data):
        corpus_folder = os.path.join(data_dir, raw_data.corpus_name)
        if not os.path.exists(corpus_folder):
            os.mkdir(corpus_folder)
        for group_name in raw_data.group_names:
            group_folder = os.path.join(corpus_folder, group_name)
            if not os.path.exists(group_folder):
                os.mkdir(group_folder)

    @staticmethod
    def create_doc_dict(raw_data):
        doc_ids = []
        for group_name in raw_data.group_names:
            doc_ids += raw_data.doc_ids(group_name)
        doc_dict = DocDict(corpus_name=raw_data.corpus_name,
                           doc_ids=doc_ids)
        doc_dict.save()

    @staticmethod
    def create_group_dict(raw_data):
        group_dict = GroupDict(corpus_name=raw_data.corpus_name,
                               group_names=raw_data.group_names)
        group_dict.save()

    @staticmethod
    def create_vocab(raw_data: RawData) -> Vocab:
        common_counts = Counter()
        with tqdm(total=len(raw_data.group_names)) as pbar:
            for group_name in raw_data.group_names:
                pbar.set_description(f'Vocab for {group_name}')

                # counts for the group
                group_counts = Counter(raw_data.group_tokens(group_name))

                # filter low frequency tokens
                group_counts = {t: c for t, c in group_counts.items()
                                if c >= raw_data.min_tok_count}

                # maintain the single vocab
                if len(common_counts) == 0:
                    common_counts.update(group_counts)
                else:
                    current_tokens = set(common_counts.keys())
                    group_tokens = set(group_counts.keys())
                    intersection = current_tokens.intersection(group_tokens)
                    common_counts = {t: c for t, c in common_counts.items()
                                     if t in intersection}
                    group_counts = {t: c for t, c in group_counts.items()
                                    if t in intersection}
                    common_counts.update(group_counts)

                pbar.update()

        # take only top-n_vocab frequent tokens
        sorted_counts = list(sorted(
            [(ix, count) for ix, count in common_counts.items()]))
        counts = sorted_counts[-raw_data.n_vocab:]
        counts = {x[0]: x[1] for x in counts}

        # create the vocab
        vocab = Vocab(raw_data.corpus_name, counts)

        # save it
        vocab.save()

        return vocab

    @staticmethod
    def file_path(corpus_name):
        return f'{data_dir}/{corpus_name}/{corpus_name}.json'

    @classmethod
    def load(cls, corpus_name: str):
        file_path = cls.file_path(corpus_name)
        with open(file_path) as f:
            data = json.loads(f.read())
            return cls(**data)

    def load_groups(self):
        groups = []
        for group_name in self.group_dict.entities:
            group = Group.load(self.name, group_name)
            groups.append(group)
        return groups

    @staticmethod
    def parse_groups_and_docs(raw_data, vocab):
        with tqdm(total=len(raw_data.group_names)) as gpbar:
            for group_name in raw_data.group_names:
                doc_ids = raw_data.doc_ids(group_name)
                docs = []
                with tqdm(total=len(doc_ids)) as dpbar:
                    for doc_id in doc_ids:
                        tokens = raw_data.doc_tokens(doc_id)
                        token_ixs = vocab.doc2ixs(tokens)
                        # subsample
                        token_ixs = Corpus.subsample(
                            token_ixs=token_ixs,
                            token_probs=vocab.probs(),
                            subsample_threshold=raw_data.subsample_threshold)
                        # create and save the doc object
                        doc = Doc(corpus_name=raw_data.corpus_name,
                                  group_name=group_name,
                                  doc_id=doc_id,
                                  token_ixs=token_ixs)
                        doc.save()
                        docs.append(doc)
                        dpbar.update()

                group = Group.from_docs(
                    corpus_name=raw_data.corpus_name,
                    name=group_name,
                    docs=docs)
                group.save()
                gpbar.update()

    def save(self):
        file_path = self.file_path(self.name)
        with open(file_path, 'w+') as f:
            data = {
                'name': self.name,
                'min_tok_count': self.min_tok_count,
                'n_vocab': self.n_vocab,
                'subsample_threshold': self.subsample_threshold,
            }
            f.write(json.dumps(data))

    @staticmethod
    def subsample(token_ixs: Sequence[int], token_probs: np.array,
                  subsample_threshold: float) \
            -> Sequence[int]:
        subsampled = []
        for ix in token_ixs:
            prob = token_probs[ix]
            p_draw = np.random.uniform(0, 1, 1)[0]
            p_threshold = 1 - np.sqrt(subsample_threshold/prob)
            keep = p_draw > p_threshold
            if keep:
                subsampled.append(ix)
        return subsampled
