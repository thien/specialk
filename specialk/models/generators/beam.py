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

    More detailed explanation of the class, its purpose,
    and any important information about its usage.

    Attributes:
        model (type): Description of field1.
        tokenizer (type): Description of field2.
        logprob_sums (type): Description of field2.
        tokens (type): Description of field2.
        ...
    """

    model: T
    tokenizer: Vocabulary
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums: Tensor, tokens: Tensor) -> Beam:
        """Creates a new Beam object with the same model and tokenizer."""
        return Beam(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> Beam:
        """Allows you to take a slice of the beams object along the batch dimension."""
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

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

        return self.new_beams(new_logprob_sums, new_tokens)

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

    def filter(self, num_beams: int) -> Tuple[Beam, Beam]:
        """Filter beams based on termination condition.

        Returns:
            (Beam): filtered version of self, containing all best `num_beams`
                which are also not terminated.
            (Beam): filtered version of self, containing all best `num_beams`
                which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        """
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
        """Get top-k logprobs ensuring no ngram repetitions."""
        if (
            no_repeat_ngram_size is None
            or self.tokens.shape[1] <= no_repeat_ngram_size - 1
        ):
            return logprobs.topk(k=k, dim=-1)

        batch, seq_len = self.tokens.shape
        neg_inf = torch.tensor(-1.0e4, device=logprobs.device)

        last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
        for i in range(seq_len - (no_repeat_ngram_size - 1)):
            ngrams = self.tokens[:, i : i + no_repeat_ngram_size]
            ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1)
            ngram_end_tokens = ngrams[:, [-1]]
            logprobs[range(batch), ngram_end_tokens] = torch.where(
                ngrams_are_repeated,
                neg_inf,
                logprobs[range(batch), ngram_end_tokens],
            )

        return logprobs.topk(k=k, dim=-1)

    @torch.inference_mode()
    def get_logits(self) -> Tensor:
        """This method should be implemented by subclasses."""
        raise NotImplementedError
