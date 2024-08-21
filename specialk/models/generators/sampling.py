from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from specialk.core import constants, log
from specialk.models.generators.beam import Beam
from specialk.models.mt_model import NMTModule
from specialk.models.tokenizer import Vocabulary


class EncoderDecoderSampler:
    """Operations to perform text sampling from an encoder/decoder model."""

    def __init__(
        self,
        model: NMTModule,
        src_tokenizer: Vocabulary,
        tgt_tokenizer: Optional[Vocabulary] = None,
        device: Optional[str] = "cpu",
    ):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer if tgt_tokenizer else src_tokenizer
        self.device = device

    @torch.inference_mode()
    def sample(
        self,
        input: str,
        max_len=50,
        start_symbol: int = constants.SOS,
        **kwargs,
    ):
        """
        Returns a string of autoregressively generated text from the decoder.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.

        args:
            input_tokens (Int[Tensor, "batch_size seq_length"]): input sequence to send to the MT model.
            max_len (int): maximum length of generated sequence.
        """
        input_tokens: Int[Tensor, "batch seq_len"] = (
            self.src_tokenizer.to_tensor(input).unsqueeze(0).to(self.device)
        )
        batch_size, x_length = input_tokens.shape
        logits: Float[Tensor, "seq_len d_vocab"]

        # setup y output values.
        y_hat = torch.full(
            (batch_size, max_len), constants.PAD, dtype=torch.long, device=self.device
        )
        y_hat[:, 0] = start_symbol

        # create encoder values
        x_mask = torch.zeros((x_length, x_length), device=self.device, dtype=torch.bool)
        x_pad_mask = self.model.create_pad_mask(input_tokens)
        x_emb = self.model.pos_encoder(
            self.model.input_emb(input_tokens) * np.sqrt(self.model.dim_model)
        )
        memory = self.model.encoder(x_emb, mask=x_mask, src_key_padding_mask=x_pad_mask)

        for i in range(1, max_len):
            # prepare decoder output to feed into the model.
            y = y_hat[:, :i]
            y_mask = self.model.generate_square_subsequent_mask(i)
            y_padding_mask = self.model.create_pad_mask(y)
            y_emb = self.model.pos_encoder(
                self.model.output_emb(y) * np.sqrt(self.model.dim_model)
            )

            out = self.model.decoder(
                tgt=y_emb,
                memory=memory,
                tgt_mask=y_mask,
                tgt_key_padding_mask=y_padding_mask,
                memory_key_padding_mask=x_pad_mask,
                tgt_is_causal=True,
            )
            logits = torch.nn.functional.log_softmax(
                self.model.generator(out[:, -1]), dim=-1
            )

            # TODO make this only 1D, but support batches later.
            logits = logits.squeeze(0)
            _y_hat = y_hat.squeeze(0)
            next_token: Tensor = self.sample_next_token(_y_hat, logits, **kwargs)
            next_token = Tensor([next_token]).to(self.device).long()
            next_token = next_token.unsqueeze(0)

            y_hat[:, i] = next_token[:, 0]
            if next_token == self.tgt_tokenizer.EOS:
                break

        # convert tokens back into strings.
        return self.tgt_tokenizer.detokenize(y_hat, specials=False)

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

        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (
            top_p != 0 and top_k != 0
        ), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return EncoderDecoderSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = EncoderDecoderSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = EncoderDecoderSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return EncoderDecoderSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return EncoderDecoderSampler.sample_top_p(logits, top_p)
        return EncoderDecoderSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        out = int(logits.argmax().item())
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
        freqs = torch.bincount(input_ids, minlength=logits.shape[-1]) * freq_penalty
        return logits - freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sample = torch.distributions.categorical.Categorical(logits=logits).sample()
        return sample

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
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
