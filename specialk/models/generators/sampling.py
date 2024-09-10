from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from specialk.core import constants, log
from specialk.models.generators.beam import Beam
from specialk.models.mt_model import NMTModule
from specialk.models.tokenizer import Vocabulary

NEG_INF = float("-inf")


class LanguageModelSampler:
    """Operations to perform text sampling from an encoder/decoder model."""

    def __init__(
        self,
        model: Optional[NMTModule] = None,
        src_tokenizer: Optional[Vocabulary] = None,
        tgt_tokenizer: Optional[Vocabulary] = None,
        device: Optional[str] = "cpu",
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer if tgt_tokenizer else src_tokenizer
        self.device = device

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "seq_len d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ) -> int:
        """Sample the next token."""

        assert input_ids.ndim == 2, "input_ids should be a 2D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"

        # Set random seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return LanguageModelSampler.greedy_search(logits)
        elif temperature != 1.0:
            # i could separate this out, but it would just do this.
            logits = logits / temperature
        if frequency_penalty != 0.0:
            logits = LanguageModelSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0 or top_p > 0.0:
            return LanguageModelSampler.sample_top_k_top_p(logits, top_k, top_p)
        return LanguageModelSampler.sample_basic(logits)

    @staticmethod
    def sample_top_k_top_p(
        logits: Float[Tensor, "batch vocab"],
        top_k: int,
        top_p: float,
        min_tokens_to_keep: int = 1,
    ) -> int:
        """
        Samples using both top-k and top-p filtering.
        """
        _, vocab_size = logits.shape

        # Apply top-k sampling; this is so we can dramatically reduce the number of tokens to
        # apply top-p for.
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), vocab_size)
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) sampling.
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_rm = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_rm[..., 1:] = sorted_indices_to_rm[..., :-1].clone()
            sorted_indices_to_rm[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_rm.scatter(
                1, sorted_indices, sorted_indices_to_rm
            )
            logits = logits.masked_fill(indices_to_remove, NEG_INF)

        # Sample from the filtered distribution
        return LanguageModelSampler.sample_basic(logits)

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> Tensor:
        """
        Samples from the top k most likely tokens.
        """
        if len(logits.shape) > 1:
            logits = logits.squeeze(0)
        topk = torch.topk(input=logits, k=k)
        top_k_logits, top_k_token_ids = topk.values, topk.indices
        sampled_token_idx = torch.distributions.categorical.Categorical(
            logits=top_k_logits
        ).sample()
        return top_k_token_ids[sampled_token_idx]

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        if len(logits.shape) > 1 and logits.shape[0] == 1:
            logits = logits.squeeze(0)
        values, _logits = logits.sort(dim=-1, descending=True, stable=True)
        cum_vals = values.softmax(dim=-1).cumsum(dim=-1)

        # first item to cut off is included in the thresholding.
        top_n = int((cum_vals <= top_p).sum() + 1)
        top_n = max(top_n, min_tokens_to_keep)

        good_logits = _logits[:top_n]
        option = np.random.choice(good_logits.cpu())
        return option

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        out = int(logits.argmax().item())
        return out

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        freq_penalty: float,
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        # TODO: this is really inefficient, needs optimising.
        freqs = torch.stack(
            [torch.bincount(seq, minlength=logits.shape[-1]) for seq in input_ids]
        )
        return logits - (freqs * freq_penalty)

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> Tensor:
        """
        Samples from the distribution defined by the logits.

        Returns:
            Tensor: one logit index from the logits.
        """
        return torch.distributions.categorical.Categorical(logits=logits).sample()

    @torch.inference_mode()
    def beam_search(
        self: LanguageModelSampler,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: Optional[int] = None,
        verbose=False,
    ) -> List[Tuple[float, Tensor]]:
        """
        Beam Search.

        This is done by repeatedly performing the `generate` and `filter` steps (starting
        from the initial prompt) until either of the two stopping criteria are met:

            (1) we've generated `max_new_tokens` tokens, or
            (2) we've generated `num_returns_sequences` terminating sequences.

        To modularize this function, most of the actual complexity is in the Beam class,
        in the `generate` and `filter` methods.
        """

        assert num_return_sequences <= num_beams
        tokens = self.tokenizer.to_tensor(prompt).to(self.device)

        # keep track of final beams; early terminations.
        beam_results: List[Tuple[float, str]] = []
        # generate logprob of prompt tokens.
        logprob_sums = torch.tensor([-1.0] * len(tokens)).to(self.device)
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
            best_beams, early_terminated_beams = best_beam.filter(num_beams)

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
