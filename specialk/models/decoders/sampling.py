from typing import List, Tuple

import torch
from jaxtyping import Float, Int
from torch import tensor as Tensor

from specialk.models.decoders.beam import Beam
from specialk.models.mt_model import NMTModule
from specialk.models.tokenizer import Vocabulary


class DecoderSampler:
    """Operations to perform text sampling from a decoder model."""

    def __init__(self, model: NMTModule, tokenizer: Vocabulary):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.generator.weights.device

    @torch.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs):
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        """
        tokens: Int[Tensor, "seq_len"]
        logits: Float[Tensor, "seq_len d_vocab"]

        tokens: List[str] = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )[0]

        for _ in range(max_tokens_generated):
            logits = self.model(tokens.unsqueeze(0))[:, -1, :]
            next_token: int = self.sample_next_token(tokens, logits, **kwargs)
            next_token = Tensor([next_token]).to(self.device).long()

            tokens = t.cat((tokens, next_token))
            if next_token == tokenizer.eos_token_id:
                break

        # convert tokens back into strings.
        return tokenizer.decode(tokens)

    @torch.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose=False,
    ) -> List[Tuple[float, t.Tensor]]:
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
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

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

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "seq_len d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (
            top_p != 0 and top_k != 0
        ), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        out = logits.argmax().item()
        return out

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        freq_penalty: float,
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        freqs = t.bincount(input_ids, minlength=logits.shape[-1]) * freq_penalty
        return logits - freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sample = t.distributions.categorical.Categorical(logits=logits).sample()
        return sample

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        if len(logits.shape) > 1:
            logits = logits.squeeze(0)
        topk = t.topk(input=logits, k=k)
        top_k_logits, top_k_token_ids = topk.values, topk.indices
        sampled_token_idx = t.distributions.categorical.Categorical(
            logits=top_k_logits
        ).sample()
        return top_k_token_ids[sampled_token_idx].item()

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
        top_n = (cum_vals <= top_p).sum() + 1
        top_n = max(top_n, min_tokens_to_keep)

        good_logits = _logits[:top_n]
        option = np.random.choice(good_logits.cpu())
        return option
