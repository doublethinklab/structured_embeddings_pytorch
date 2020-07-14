"""Structured Bernoulli Embedding Models.

Pytorch implementation of:
https://github.com/mariru/structured_embeddings
https://papers.nips.cc/paper/6629-structured-embedding-models-for-grouped-data
"""
from typing import Optional

import torch
from torch import distributions, nn


class Batch:

    def __init__(self):
        pass


class HierarchicalBernoulliEmbeddings(nn.Module):

    def __init__(self, context_size: int, negative_samples: int, n_vocab: int,
                 n_dim: int, n_states: int, unigram_logits: torch.Tensor,
                 sigma: float):
        """Create a new Hierarchical Bernoulli model.

        Args:
          context_size: Int, size of the context window.
          negative_samples: Int, number of negative samples.
          n_vocab: Int, number of tokens in the vocabulary.
          n_dim: Int, size of embedding vectors.
          n_states: Int, number of specific state embeddings.
          unigram_logits: torch.Tensor, unnormalized log probabilities for each
            token in the vocab.
          sigma: Float, variance of Gaussian regularizing the global embedding
            matrices.
        """
        super().__init__()
        if context_size % 2 != 0:
            raise ValueError(f'Context size {context_size} not divisible by 2.')
        self.cs = context_size
        self.ns = negative_samples
        self.sigma = sigma
        self.rho = self.init_embeddings(n_vocab, n_dim, sigma)
        self.alpha = self.init_embeddings(n_vocab, n_dim, sigma)
        self.rho_state = {}
        self.n_states = n_states
        for i in range(n_states):
            self.rho_state = self.init_embeddings(n_vocab, n_dim, sigma=0.0001,
                                                  weight=self.rho.weight)
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
        # for each state
        #   calculate the bernoulli dist probs for +ve and -ve samples
        #   calculate the loss and return

        # predictions for each state
        preds = {}
        for s, state in enumerate(batch.states):

            # NOTE: it looks like they use the same context, and negative
            #  samples for the rhos, not the alphas.
            #  main question now is: how to form up a batch of data?

            # selecting data points
            p_mask = torch.range(self.cs / 2, n + self.cs / 2)

            n_idx = torch.multinomial(self.unigram_logits, self.ns)

            ll_neg = None
            loss += - (ll_pos + ll_neg + lp_prior)

    def loss(self) -> torch.Tensor:
        """Calculate the loss term.

        .. math::
          \\displaymath \mathcal{L}_{\text{hier}} =
                            \log p(\alpha)
                            + \log p(\rho^{(0)})
                            + \sum_s \log p(\rho^{(s)}|\rho^{(0)})
                            + \sum_{v,i} \log p(x_{vi}|x_{c_{vi}}; \alpha, \rho)

        Returns:
          torch.Tensor.  (or Variable?)
        """
        # regularization
        prior = distributions.Normal(loc=0., scale=self.sigma)
        loss = prior.log_prob(self.rho)
        loss = loss + prior.log_prob(self.alpha)
        local_prior = distributions.Normal(loc=0., scale=self.sigma/100.)
        for i in range(self.n_states):
            # in their code that conditional is this diff
            # https://github.com/mariru/structured_embeddings/blob/master/src/models.py#L326
            diff = self.rho - self.rho_state[i]
            loss = loss + local_prior.log_prob(diff)
