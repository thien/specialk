from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from specialk.models.mt_model import NMTModule
from specialk.models.tokenizer import Vocabulary
from specialk.core import log
from rich.table import Table
from rich import print as rprint

@dataclass
class Beam:
    """Class to store beams during beam search."""

    model: NMTModule 
    tokenizer: Vocabulary 
    logprob_sums: Float[Tensor, "batch"]  # each item corresponds to index 0 of tokens.
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums, tokens) -> Beam:
        """Creates a new Beam object with the same model and tokenizer."""
        return Beam(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> Beam:
        """Allows you to take a slice of the beams object along the batch dimension."""
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, str]]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(
        self, toks_per_beam: int, no_repeat_ngram_size: Optional[int] = None
    ) -> Beam:
        """
        Arguments:
            toks_per_beam: beam size.
            no_repeat_ngram_size (int, Optional): if set, determines n-gram for duplication killing.
        Returns:
            Beam: Beam of next tokens with the new tokens.

        Starting from the current set of beams (which has length `num_beams`),
        returns a new set of `num_beams * toks_per_beam`, containing the best
        `toks_per_beam` continuations for each of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate
        any sequences with a repeating n-gram of this length.
        """
        logits: Float[Tensor, "seq_len d_vocab"]

        # generate dist of next tokens.
        with torch.no_grad():
            logits = self.model(self.tokens)[:, -1, :]
            log_logits = logits.log_softmax(-1)

        # sample top-k.
        topk_log_probs: Float[Tensor, "batch toks_per_beam"]
        topk_token_idx: Int[Tensor, "batch toks_per_beam"]
        topk_log_probs, topk_token_idx = self.get_topk_non_repeating(
            log_logits, no_repeat_ngram_size, k=toks_per_beam
        )

        # Calculate new log probabilities.
        # Below, we increase the size of logprob_sums by logprob_sums*toks_per_beam.
        # so we need to add the generated log prob to each item.
        new_logprob_sums = sum(
            [
                einops.repeat(self.logprob_sums, "batch -> (batch k)", k=toks_per_beam),
                einops.rearrange(topk_log_probs, "batch k -> (batch k)"),
            ]
        )

        new_tokens = torch.concat(
            [
                einops.repeat(
                    self.tokens, "batch seq -> (batch k) seq", k=toks_per_beam
                ),
                einops.rearrange(topk_token_idx, "batch k -> (batch k) 1"),
            ],
            dim=-1,
        ).long()

        return self.new_beams(new_logprob_sums, new_tokens)

    def filter(self, num_beams: int) -> Tuple["Beam", "Beam"]:
        """
        Returns:
            best_beams: Beam
                filtered version of self, containing all best `num_beams` which are
                also not terminated.

            early_terminations: Beam
                filtered version of self, containing all best `num_beams` which are
                also terminated. i.e. the sum of lengths of these two should equal
                `num_beams`.
        """
        # early termination is caused by predicting the EOS token.
        EOS: int = self.tokenizer.eos_token_id
        # split self.tokens based on whether EOS is predicted or notorch.
        idx_terminated = (self.tokens == EOS).any(dim=-1)
        return self[~idx_terminated], self[idx_terminated]

    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logitsums.
        """
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)

    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: Optional[int],
        k: int,
    ) -> Tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        logprobs:
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure
            that no returned tokens would produce an ngram of size  `no_repeat_ngram_size`
            which has already appeared in `self.tokens`.
        """
        batch, seq_len = self.tokens.shape
        neg_inf = torch.tensor(-1.0e4).to(logprobs.device)

        # If completion isn't long enough for a repetition, or we have no restructions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size - 1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            # Next, find all the tokens we're not allowed to generate (by going iterating through past
            # ngrams and seeing if those ngram prefixes match the last one)
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]  # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(
                    -1
                )  # (batch,)
                ngram_end_tokens = ngrams[:, [-1]]  # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = torch.where(
                    ngrams_are_repeated,
                    neg_inf,
                    logprobs[range(batch), ngram_end_tokens],
                )

        # Finally, get our actual tokens
        return logprobs.topk(k=k, dim=-1)

    def __repr__(self) -> str:
        return (
            f"Beam(model={self.model.__class__.__name__}, "
            f"tokenizer={self.tokenizer.__class__.__name__}, "
            f"logprob_sums={self.logprob_sums.shape}, "
            f"tokens={self.tokens.shape})"
        )


@torch.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False,
) -> List[Tuple[float, Tensor]]:
    """
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beam class,
    in the `generate` and `filter` methods.
    """

    assert num_return_sequences <= num_beams
    self.model.eval()
    tokens = self.tokenizer.to_tensor(prompt).to(device)

    # keep track of final beams; early terminations.
    beam_results: List[Tuple[float, str]] = []
    # generate logprob of prompt tokens.
    logprob_sums = torch.tensor([-1.0] * len(tokens)).to(device)
    # logprob_sums = self.model(tokens)[0].log_softmax(-1).diagonal()[-2:-1]
    print(logprob_sums.shape, logprob_sums)
    best_beam = Beam(
        model=self.model,
        tokenizer=self.tokenizer,
        logprob_sums=logprob_sums,
        tokens=tokens,
    )
    for i in tqdm(range(max_new_tokens)):
        # generate beam.
        best_beam = best_beam.generate(
            num_beams, no_repeat_ngram_size=no_repeat_ngram_size
        )
        best_beam, early_terminated_beams = best_beam.filter(num_beams)

        beam_results.extend(early_terminated_beams.logprobs_and_completions)

        if verbose:
            best_beams.print(title=f"Best Completions @ idx={i}")
            early_terminated_beams.print(
                title=f"Early Terminated Completions @ idx={i}"
            )

        # early stopping condition.
        if len(beam_results) >= num_return_sequences:
            return beam_results[:num_return_sequences]

    beam_results.extend(best_beams.logprobs_and_completions)
    beam_results = beam_results[:num_return_sequences]
    return beam_results
