"""Structured Bernoulli Embedding Models.

Pytorch implementation of:
https://github.com/mariru/structured_embeddings
https://papers.nips.cc/paper/6629-structured-embedding-models-for-grouped-data
"""
from typing import Optional, Sequence

import torch
from torch import distributions, nn


class StateBatch:
    """Batch for a state.

    NOTE: as opposed to the original TensorFlow implementation, masks for
      vector selection are done by the DataLoader in this PyTorch implementation
      to exploit parallelization.
    """

    def __init__(self, target_ixs: torch.LongTensor,
                 context_ixs: torch.LongTensor,
                 negative_sample_ixs: torch.LongTensor):
        self.target_ixs = target_ixs
        self.context_ixs = context_ixs
        self.negative_sample_ixs = negative_sample_ixs


class Batch:
    """Full batch object.

    Wraps the state batches.
    """

    def __init__(self, states: Sequence[StateBatch]):
        self.states = states


class HierarchicalBernoulliEmbeddings(nn.Module):

    def __init__(self, n_context: int, n_negative_samples: int, n_vocab: int,
                 n_dim: int, n_states: int, unigram_logits: torch.Tensor,
                 sigma: float):
        """Create a new Hierarchical Bernoulli model.

        Args:
          n_context: Int, size of the context window.
          n_negative_samples: Int, number of negative samples.
          n_vocab: Int, number of tokens in the vocabulary.
          n_dim: Int, size of embedding vectors.
          n_states: Int, number of specific state embeddings.
          unigram_logits: torch.Tensor, unnormalized log probabilities for each
            token in the vocab.
          sigma: Float, variance of Gaussian regularizing the global embedding
            matrices.
        """
        super().__init__()
        if n_context % 2 != 0:
            raise ValueError(f'Context size {n_context} not divisible by 2.')
        self.cs = n_context
        self.ns = n_negative_samples
        self.sigma = sigma
        self.word_embeds = self.init_embeddings(n_vocab, n_dim, sigma)
        self.context_embeds = self.init_embeddings(n_vocab, n_dim, sigma)
        self.rho_state = {}
        self.n_states = n_states
        for i in range(n_states):
            self.rho_state = self.init_embeddings(
                n_vocab, n_dim, sigma=0.0001, weight=self.word_embeds.weight)
        self.unigram_logits = unigram_logits

    @staticmethod
    def init_embeddings(n_vocab: int, n_dim: int, sigma: float,
                        weight: Optional[nn.Parameter] = None) -> nn.Embedding:
        """Initialize an embedding matrix.

        Args:
          n_vocab: Int, number of items in the vocab.
          n_dim: Int, size of embedding vectors.
          sigma: Float, scaling for initialization. The pytorch default is a
            random normal with unit variance. In the structured embeddings
            paper code, the global embeddings are initialized with 0.1 sigma.
          weight: torch.nn.Parameter, optional weight initialization matrix,
            used to initialize an embedding down in the hierarchy with its
            parent.
        """
        if not weight:
            weight = nn.Parameter(torch.randn((n_vocab, n_dim)))
        weight = weight * sigma
        return nn.Embedding(n_vocab, n_dim, _weight=weight)

    def forward(self, batch: Batch):
        # NOTE: since the states can yield different length texts, do it their
        #  way by for loop over states, as opposed to forcing same lengths into
        #  a single Tensor.
        # TODO: think through bootstrapping and subsampling

        # predictions for each state
        logits = []
        for state in batch.states:
            # [batch, n_dim]
            targets = self.word_embeds(state.target_ixs)
            # [batch, n_dim]
            contexts = self.context_embeds(state.context_ixs).sum(dim=1)
            # [batch, n_dim]
            negative_samples = self.word_embeds(state.negative_sample_ixs)

            # logits: [batch]
            positive_logits = (targets * contexts).sum(dim=1)
            negative_logits = (negative_samples * contexts).sum(dim=1)

            logits.append((positive_logits, negative_logits))

        loss = self.loss(logits)

        return loss

    def loss(self, logits: Sequence[(torch.Tensor, torch.Tensor)]) \
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
        loss = prior.log_prob(self.word_embeds)
        loss = loss + prior.log_prob(self.context_embeds)
        local_prior = distributions.Normal(loc=0., scale=self.sigma/100.)
        for i in range(self.n_states):
            # in their code that conditional is this diff
            # https://github.com/mariru/structured_embeddings/blob/master/src/models.py#L326
            diff = self.word_embeds - self.rho_state[i]
            loss = loss + local_prior.log_prob(diff)
            p_logits, n_logits = logits[i]
            p_dist = torch.distributions.Bernoulli(logits=p_logits)
            n_dist = torch.distributions.Bernoulli(logits=n_logits)
            p_logloss = p_dist.log_prob(1.)
            n_logloss = n_dist.log_prob(0.)
            loss = loss + p_logloss
            loss = loss + n_logloss
        return loss
