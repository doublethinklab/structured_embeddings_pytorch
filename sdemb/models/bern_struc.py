"""Structured Bernoulli Embedding Models.

Pytorch implementation of:
https://github.com/mariru/structured_embeddings
https://papers.nips.cc/paper/6629-structured-embedding-models-for-grouped-data
"""
import collections
import json
import math
import types
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import tensorboardX
import torch
from torch import distributions, nn, optim
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset as TorchDataset
from tqdm.notebook import tqdm

from sdemb import data


class Dataset(TorchDataset):
    """Dataset wrapper."""

    def __init__(self, corpus: data.Corpus, subset: str, n_batches: int):
        self.corpus = corpus
        self.subset = subset
        self.n_batches = n_batches
        self.n_groups = corpus.n_groups
        self.n_tokens_per_group = [g.n_tokens[subset] for g in corpus.groups]
        self.n_tokens_per_batch = [int(math.ceil(n / n_batches))
                                   for n in self.n_tokens_per_group]

    def __getitem__(self, item) -> Sequence[torch.LongTensor]:
        # `item` is meaningless, this function always yields the next batch
        batch_data = []
        for ix in range(self.n_groups):
            n_tokens_to_take = self.n_tokens_per_batch[ix]
            token_ixs = self.corpus.groups[ix].get_batch(
                self.subset, n_tokens_to_take)
            token_ixs = torch.LongTensor(token_ixs)
            batch_data.append(token_ixs)
        return batch_data

    def __len__(self):
        return self.n_batches


class GroupBatch:
    """Batch for a specific corpus.

    NOTE: as opposed to the original TensorFlow implementation, masks for
      vector selection are done by the DataLoader in this PyTorch implementation
      to exploit parallelization.
    """

    def __init__(self, target_ixs: torch.Tensor, context_ixs: torch.Tensor,
                 negative_sample_ixs: torch.Tensor):
        self.target_ixs = target_ixs
        self.context_ixs = context_ixs
        self.negative_sample_ixs = negative_sample_ixs

    def to_device(self, device):
        self.target_ixs.to(device)
        self.context_ixs.to(device)
        self.negative_sample_ixs.to(device)


class Collate:
    """Collate function."""

    def __init__(self, n_context: int, token_probs: np.array,
                 n_negative_samples: int):
        """Create a new Collate function.

        Args:
          n_context: Int, number of words in context window.
          token_probs: np.array, unnormalized log probabilities for each
            token in the vocab.
          n_negative_samples: Int, number of negative samples to draw.
        """
        self.n_context = n_context
        self.sample_probs = self.get_sample_probs(token_probs)
        self.n_negative_samples = n_negative_samples

    def __call__(self, items: Sequence[Sequence[torch.Tensor]]) \
            -> Sequence[GroupBatch]:
        items = items[0]  # using DataLoader wraps batch in a dim
        corpus_batches = []
        for token_ixs in items:
            target_ixs = self.get_target_ixs(token_ixs)
            context_ixs = self.get_context_ixs(token_ixs)
            negative_sample_ixs = self.get_negative_sample_ixs(token_ixs)
            corpus_batch = GroupBatch(
                target_ixs, context_ixs, negative_sample_ixs)
            corpus_batches.append(corpus_batch)
        return corpus_batches

    def get_target_ixs(self, token_ixs: torch.Tensor) -> torch.Tensor:
        # subtract the context window in order to leave room either side
        n_targets = token_ixs.shape[0] - self.n_context
        target_mask = torch.arange(
            int(self.n_context/2), n_targets+int(self.n_context/2))
        return token_ixs[target_mask]

    def get_context_ixs(self, token_ixs: torch.Tensor) -> torch.Tensor:
        n_targets = token_ixs.shape[0] - self.n_context
        hc = int(self.n_context / 2)  # half-context (i.e. one-sided)
        rows = torch.arange(0, hc).unsqueeze(0).repeat([n_targets, 1])
        cols = torch.arange(0, n_targets).unsqueeze(1).repeat(1, hc)
        context_mask = torch.cat([rows + cols, rows + cols + hc + 1], dim=1)
        return token_ixs[context_mask]

    def get_negative_sample_ixs(self, token_ixs: torch.Tensor) -> torch.Tensor:
        n_targets = token_ixs.shape[0] - self.n_context
        return torch.multinomial(self.sample_probs.repeat([n_targets, 1]),
                                 self.n_negative_samples)

    @staticmethod
    def get_sample_probs(token_probs):
        return torch.from_numpy(
            token_probs ** (3. / 4.)
            / (token_probs ** (3. / 4.)).sum()).float()


class HierarchicalBernoulliEmbeddings(nn.Module):

    def __init__(self, n_context: int, n_negative_samples: int, n_vocab: int,
                 n_dim: int, n_groups: int, sigma: float,
                 initial_embeds: Optional[np.ndarray] = None):
        """Create a new Hierarchical Bernoulli model.

        Args:
          n_context: Int, size of the context window.
          n_negative_samples: Int, number of negative samples.
          n_vocab: Int, number of items in the vocabulary.
          n_dim: Int, size of embedding vectors.
          n_groups: Int, number of specific group embeddings.
          sigma: Float, variance of Gaussian regularizing the global embedding
            matrices.
        """
        super().__init__()
        if n_context % 2 != 0:
            raise ValueError(f'Context size {n_context} not divisible by 2.')
        self.n_context = n_context
        self.n_negative_samples = n_negative_samples
        self.n_vocab = n_vocab
        self.n_dim = n_dim
        self.n_groups = n_groups
        self.sigma = sigma
        # TODO: gensim didn't train context embeds, so back to square one
        self.word_embeds = self.init_embedding()
        self.context_embeds = self.init_embedding()
        self.group_embeds = []

    def init_embedding(self, weight: Optional[nn.Parameter] = None) \
            -> nn.Embedding:
        """Initialize an embedding matrix.

        Args:
          weight: torch.nn.Parameter, optional weight initialization matrix,
            used to initialize an embedding down in the hierarchy with its
            parent.
        """
        if weight is None:
            weight = nn.Parameter(torch.randn((self.n_vocab, self.n_dim)))
        weight = weight * self.sigma
        return nn.Embedding(self.n_vocab, self.n_dim, _weight=weight,
                            sparse=True)

    def init_group_embeddings(self):
        # this is separate from __init__ because we train globals first,
        # then initialize the group embeddings with the globals.
        for _ in range(self.n_groups):
            self.group_embeds.append(
                self.init_embedding(self.word_embeds.weight))

    def forward(self, batch: Sequence[GroupBatch]):
        # NOTE: since the corpus_batches can yield different length texts, do
        #  it their way by for loop over corpus_batches, as opposed to forcing
        #  same lengths into a single Tensor.
        # TODO: think through bootstrapping and subsampling

        # predictions for each state
        logits = []
        for corpus_batch in batch:
            # [batch, n_dim]
            targets = self.word_embeds(corpus_batch.target_ixs)
            # [batch, n_dim]
            contexts = self.context_embeds(corpus_batch.context_ixs).sum(dim=1)
            # [batch, n_negs, n_dim]
            negative_samples = self.word_embeds(
                corpus_batch.negative_sample_ixs)

            # logits: [batch]
            positive_logits = (targets * contexts).sum(dim=1)
            contexts = contexts.unsqueeze(1).repeat(
                [1, self.n_negative_samples, 1])
            negative_logits = (negative_samples * contexts)\
                .sum(dim=1).sum(dim=1)

            logits.append((positive_logits, negative_logits))

        loss = self.loss(logits)

        return loss

    def loss(self, logits: Sequence[tuple]) \
            -> torch.Tensor:
        """Calculate the loss term.

        .. math::
          \\displaymath \mathcal{L}_{\text{hier}} =
                            \log p(\alpha)
                            + \log p(\rho^{(0)})
                            + \sum_s \log p(\rho^{(s)}|\rho^{(0)})
                            + \sum_{v,i} \log p(x_{vi}|x_{c_{vi}}; \alpha, \rho)

        Args:
          logits: Dict, mapping Int state ixs to their torch.Tensor logits.

        Returns:
          torch.Tensor.  (or Variable?)
        """
        # regularization
        prior = distributions.Normal(loc=0., scale=self.sigma)
        loss = prior.log_prob(self.word_embeds.weight).sum()
        loss = loss + prior.log_prob(self.context_embeds.weight).sum()

        if self.group_embeds:  # only add this loss after initialized
            local_prior = distributions.Normal(loc=0., scale=self.sigma/100.)
            for i in range(self.n_groups):
                # in their code that conditional is this diff
                # https://github.com/mariru/structured_embeddings/blob/master/src/models.py#L326
                diff = (self.word_embeds.weight
                        - self.group_embeds[i].weight).sum()
                loss = loss + local_prior.log_prob(diff)
                p_logits, n_logits = logits[i]
                p_dist = torch.distributions.Bernoulli(logits=p_logits)
                n_dist = torch.distributions.Bernoulli(logits=n_logits)
                p_logloss = p_dist.log_prob(1.).sum()
                n_logloss = n_dist.log_prob(0.).sum()
                loss = loss + p_logloss
                loss = loss + n_logloss

        # gradient ascent
        loss = loss * -1

        return loss

    def fit(self, corpus: data.Corpus, lr: float, n_batches: int,
            n_epochs: int):
        train = Dataset(
            corpus=corpus,
            subset='train',
            n_batches=n_batches)
        collate = Collate(
            self.n_context, corpus.vocab.probs(), self.n_negative_samples)
        train_loader = dataloader.DataLoader(
            dataset=train,
            batch_size=1,
            collate_fn=collate,
            shuffle=False)
        optimizer = optim.Adam(params=self.parameters(), lr=lr)
        writer = tensorboardX.SummaryWriter('temp/runs')

        # 1. fit the global embeddings
        global_step = 0
        for _ in range(n_epochs):
            with tqdm(total=len(train_loader), desc='Global') as pbar:
                for batch in train_loader:
                    global_step += 1
                    loss = self.forward(batch)
                    loss.backward()
                    optimizer.step()
                    self.zero_grad()
                    writer.add_scalar(
                        'loss', float(loss.detach().cpu().numpy()), global_step)
                    pbar.update()

        # 2. fit the group embeddings
        self.init_group_embeddings()
        for _ in range(n_epochs):
            with tqdm(total=len(train_loader), desc='Groups') as pbar:
                for batch in train_loader:
                    loss = self.forward(batch)
                    loss.backward()
                    optimizer.step()
                    self.zero_grad()
                    writer.add_scalar(
                        'loss', float(loss.detach().cpu().numpy()), global_step)
                    pbar.update()

        writer.close()
