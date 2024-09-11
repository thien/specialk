from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import einops
import torch
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor

from specialk.models.tokenizer import Vocabulary

T = TypeVar("T")


@dataclass
class Beam(Generic[T]):
    """Class to store Beams for Beam Search.

    Attributes:
        model (T): Model used to generate logits.
        tokenizer (Vocabulary): Tokenizer to decode logits into words.
        logprob_sums (Tensor): Logprobs of the tokens.
        tokens (Tensor): Tokens generated from the model.
        ...
    """

    model: T
    tokenizer: Vocabulary
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    @property
    def num_beams(self) -> int:
        return self.logprob_sums.size(0)

    def new(self, logprob_sums: Tensor, tokens: Tensor) -> Beam:
        """Creates a new Beam object with the same model and tokenizer."""
        return Beam(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> Beam:
        """Allows you to take a slice of the beams object along the batch dimension."""
        return self.new(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, List[str]]]:
        """Returns self as a list of logprob sums and completions."""
        return [
            (logprob_sum.item(), self.tokenizer.detokenize(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    @property
    def logprobs_and_tensors(self) -> List[Tuple[float, Tensor]]:
        """Returns self as a list of logprob sums and tensors. The tensors
        will require detokeniisation."""
        return [
            (logprob_sum.item(), tokens)
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(
        self, tokens_per_beam: int, no_repeat_ngram_size: Optional[int] = None
    ) -> Beam:
        """Generate next set of beams."""
        log_logits = self.get_logits()

        topk_log_probs, topk_token_idx = self.get_topk_non_repeating(
            log_logits, no_repeat_ngram_size, k=tokens_per_beam
        )

        new_logprob_sums = self._calculate_new_logprob_sums(
            topk_log_probs, tokens_per_beam
        )
        new_tokens = self._generate_new_tokens(topk_token_idx, tokens_per_beam)

        return self.new(new_logprob_sums, new_tokens)

    def _calculate_new_logprob_sums(
        self, topk_log_probs: Tensor, tokens_per_beam: int
    ) -> Tensor:
        """Repeats the logprob_sums down the batch dimension."""
        return einops.repeat(
            self.logprob_sums, "batch -> (batch k)", k=tokens_per_beam
        ) + einops.rearrange(topk_log_probs, "batch k -> (batch k)")

    def _generate_new_tokens(
        self, topk_token_idx: Tensor, tokens_per_beam: int
    ) -> Tensor:
        return torch.cat(
            [
                einops.repeat(
                    self.tokens, "batch seq -> (batch k) seq", k=tokens_per_beam
                ),
                einops.rearrange(topk_token_idx, "batch k -> (batch k) 1"),
            ],
            dim=-1,
        ).long()

    def print(self, title: str = "Best completions", max_print_chars: int = 80) -> None:
        """Prints out a set of sequences with their corresponding logit sums."""
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.detokenize(tokens)
            if len(repr(text)) > max_print_chars:
                text = f"{text[:int(0.3 * max_print_chars)]} ... {text[-int(0.7 * max_print_chars):]}"
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)

    def filter(self, num_beams: Optional[int] = None) -> Tuple[Beam, Beam]:
        """Filter beams based on termination condition.

        Returns:
            (Beam): filtered version of self, containing all best `num_beams`
                which are also not terminated.
            (Beam): filtered version of self, containing all best `num_beams`
                which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        """
        if num_beams is None:
            num_beams = self.num_beams

        top_beam_indices = self.logprob_sums.topk(k=num_beams, dim=0).indices
        is_terminated = (self.tokens == self.tokenizer.EOS).any(dim=-1)

        top_beam_mask = torch.zeros_like(is_terminated, dtype=torch.bool)
        top_beam_mask[top_beam_indices] = True

        best_continuing_mask = top_beam_mask & ~is_terminated
        best_terminated_mask = top_beam_mask & is_terminated

        return self[best_continuing_mask], self[best_terminated_mask]

    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: Optional[int],
        k: int,
    ) -> Tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        Get top-k logprobs ensuring no ngram repetitions.

        Args:
            logprobs: Tensor of shape (batch, d_vocab) containing log probabilities.
            no_repeat_ngram_size: Size of ngrams to avoid repeating.
            k: Number of top probabilities to return.

        Returns:
            Tuple of (top-k values, top-k indices)
        """
        if self._should_skip_ngram_check(no_repeat_ngram_size):
            return logprobs.topk(k=k, dim=-1)

        _, seq_len = self.tokens.shape

        last_ngram_prefix = self._get_last_ngram_prefix(no_repeat_ngram_size)

        for i in range(seq_len - (no_repeat_ngram_size - 1)):
            ngrams = self.tokens[:, i : i + no_repeat_ngram_size]
            self._mask_repeated_ngrams(logprobs, ngrams, last_ngram_prefix)

        return logprobs.topk(k=k, dim=-1)

    def _should_skip_ngram_check(self, no_repeat_ngram_size: Optional[int]) -> bool:
        """
        Determine if the ngram repetition check should be skipped.

        Args:
            no_repeat_ngram_size: Size of ngrams to avoid repeating

        Returns:
            Boolean indicating whether to skip the ngram check
        """
        return (
            no_repeat_ngram_size is None
            or self.tokens.shape[1] <= no_repeat_ngram_size - 1
        )

    def _get_last_ngram_prefix(self, no_repeat_ngram_size: int) -> Tensor:
        """
        Get the prefix of the last ngram in the sequence.

        Args:
            no_repeat_ngram_size: Size of ngrams to avoid repeating

        Returns:
            Tensor containing the prefix of the last ngram
        """
        return self.tokens[:, -no_repeat_ngram_size + 1 :]

    def _mask_repeated_ngrams(
        self,
        logprobs: Tensor,
        ngrams: Tensor,
        last_ngram_prefix: Tensor,
        neg_inf: Tensor = torch.tensor(-1.0e4),
    ) -> None:
        """
        Mask the logprobs of repeated ngrams with negative infinity.

        Args:
            logprobs: Tensor of shape (batch, d_vocab) containing log probabilities
            ngrams: Tensor of current ngrams being checked
            last_ngram_prefix: Tensor containing the prefix of the last ngram
            neg_inf (Optional[Tensor]): Tensor with negative infinity value for masking.

        Returns:
            None (modifies logprobs in-place).
        """
        if neg_inf.device != logprobs.device:
            neg_inf = neg_inf.to(logprobs.device)

        ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1)
        ngram_end_tokens = ngrams[:, -1]

        batch_indices = torch.arange(logprobs.shape[0], device=logprobs.device)
        logprobs[batch_indices, ngram_end_tokens] = torch.where(
            ngrams_are_repeated, neg_inf, logprobs[batch_indices, ngram_end_tokens]
        )

    @torch.inference_mode()
    def get_logits(self) -> Tensor:
        """This method should be implemented by subclasses."""
        raise NotImplementedError
