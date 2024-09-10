"""
PyTorch native implementation of the Transformer.

This is intentional to take advantage of native
implementations (as reasonably as possible), so a lot of this
will be calling the relevant classes and modified afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from specialk.core.utils import log
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import LayerNorm
from torch.optim import AdamW
import einops

import specialk.core.constants as Constants
from specialk.models.generators.sampling import LanguageModelSampler
from specialk.models.mt_model import EncoderDecoderBeam, NMTModule
from specialk.models.optimizers.schedulers import CosineWarmupScheduler
from specialk.models.transformer.pos_encoders import PositionalEncoder
from specialk.models.utils.activations import SwiGLU


@dataclass
class TransformerEncoderDecoderBeam(EncoderDecoderBeam):
    """Beam Class to be used with PyTorchTransformerModel.

    Attributes:
        model (type): Description of field1.
        tokenizer (type): Description of field2.
        logprob_sums (type): Description of field2.
        tokens (type): Description of field2.
        memory (Tensor): encoder output.
        x_pad_mask (Tensor): mask tensor for input tokens.
    """

    memory: Tensor
    x_pad_mask: Tensor

    def new_beams(
        self,
        logprob_sums: Float[Tensor, "batch"],
        tokens: Int[Tensor, "batch seq"],
        memory: Tensor,
        x_pad_mask: Tensor,
    ) -> TransformerEncoderDecoderBeam:
        """Creates a new EncoderDecoderBeam object with the
        same model, tokenizer, and encoder_output."""
        return TransformerEncoderDecoderBeam(
            model=self.model,
            tokenizer=self.tokenizer,
            logprob_sums=logprob_sums,
            tokens=tokens,
            memory=memory,
            x_pad_mask=x_pad_mask,
        )

    @torch.inference_mode()
    def get_logits(self) -> Tensor:
        logits = self.model.decode(
            self.tokens,
            tgt_mask=None,
            memory=self.memory,
            x_pad_mask=self.x_pad_mask,
        )

        logits = torch.nn.functional.log_softmax(
            self.model.generator(logits[:, -1]), dim=-1
        )
        return logits

    def expand_encoder_outputs(self, tokens_per_beam: int) -> Tensor:
        return einops.repeat(
            self.memory,
            "batch enc_seq hidden -> (batch beam) enc_seq hidden",
            beam=tokens_per_beam,
        )

    def expand_x_pad_masks(self, tokens_per_beam: int) -> Tensor:
        return einops.repeat(
            self.x_pad_mask,
            "batch enc_seq -> (batch beam) enc_seq",
            beam=tokens_per_beam,
        )

    def __getitem__(self, idx) -> TransformerEncoderDecoderBeam:
        """Allows you to take a slice of the beams object along the batch dimension."""
        return self.new_beams(
            self.logprob_sums[idx],
            self.tokens[idx],
            self.memory[idx],
            self.x_pad_mask[idx],
        )

    def generate(
        self, tokens_per_beam: int, no_repeat_ngram_size: Optional[int] = None
    ) -> "Beam":
        log_logits = self.get_logits()

        topk_log_probs, topk_token_idx = self.get_topk_non_repeating(
            log_logits, no_repeat_ngram_size, k=tokens_per_beam
        )

        new_logprob_sums = self._calculate_new_logprob_sums(
            topk_log_probs, tokens_per_beam
        )
        new_tokens = self._generate_new_tokens(topk_token_idx, tokens_per_beam)
        new_encoder_outputs = self.expand_encoder_outputs(tokens_per_beam)
        new_x_pad_mask = self.expand_x_pad_masks(tokens_per_beam)
        return self.new_beams(
            new_logprob_sums, new_tokens, new_encoder_outputs, new_x_pad_mask
        )


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Modified Transformer Encoder Layer so we can add optional SwiGLU."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )

        if activation == SwiGLU:
            self.linear1 = SwiGLU(d_model, dim_feedforward, d_model, bias=bias)
            self.linear2 = nn.Identity()


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """Modified Transformer Decoder Layer so we can add optional SwiGLU."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )
        if activation == SwiGLU:
            self.linear1 = SwiGLU(d_model, dim_feedforward, d_model, bias=bias)
            self.linear2 = nn.Identity()


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        super().__init__(
            encoder_layer, num_layers, norm, enable_nested_tensor, mask_check
        )


class PyTorchTransformerModel(nn.Transformer):
    def __init__(
        self,
        vocab_size: int,
        decoder_vocab_size: Optional[int] = None,
        max_seq_length: int = 100,
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout=0.1,
        decoder_generator_weight_sharing=True,
        name: str = "PyTorchTransformer",
        activation: Callable = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """PyTorch native implementation of a Transformer (see parent class).

        Args:
            vocab_size (int): Vocabulary size of tokenizer.
            max_seq_length (int, optional): Maximum sequence length.
                Defaults to 100.
            dim_model (int, optional): The number of expected features in the
                encoder/decoder inputs. Defaults to 512.
            n_heads (int, optional): The number of self-attention heads.
                Defaults to 8.
            dim_feedforward (int, optional): Dimension of the FFM. Defaults to 2048.
            num_encoder_layers (int, optional): Number of attn layers in the encoder.
                Defaults to 6.
            num_decoder_layers (int, optional): Number of attn layers in the decoder.
                Defaults to 6.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            decoder_generator_weight_sharing (bool, optional): If set, shares weight
                between deocder and generator. Defaults to True.
        """
        # initialize encoder and decoders separately.
        factory_kwargs = {"device": device, "dtype": dtype}
        encoder_layer = TransformerEncoderLayer(
            dim_model,
            n_heads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )

        encoder_norm = LayerNorm(
            dim_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            dim_model,
            n_heads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            **factory_kwargs,
        )
        decoder_norm = LayerNorm(
            dim_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        super(PyTorchTransformerModel, self).__init__(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=batch_first,
            custom_encoder=encoder,
            custom_decoder=decoder,
        )
        self.name = name
        self.model_type = "Transformer"
        self.max_seq_length = max_seq_length
        self.dim_model = dim_model
        self.tgt_mask = None
        self.decoder_generator_weight_sharing = decoder_generator_weight_sharing
        self.x_logit_scale = 1.0
        self.dropout = dropout
        if not decoder_vocab_size:
            decoder_vocab_size = vocab_size

        self.input_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim_model
        )
        self.output_emb = nn.Embedding(
            num_embeddings=decoder_vocab_size, embedding_dim=dim_model
        )
        self.pos_encoder = PositionalEncoder(
            dim_model=dim_model, max_seq_length=max_seq_length
        )
        self.generator = nn.Linear(dim_model, decoder_vocab_size)
        self.init_weights()

    def generate_square_subsequent_mask(self, size: int) -> Tensor:
        """Generate square causal mask.

        Top half of the diagonal is -inf, else 0's.

        Parameters:
                length (int): Number of tokens in each sequence in the target batch.
        Returns:
                mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                  [0.,   0., -inf],
                                                  [0.,   0.,   0.]]
                            for a size=3.
        """
        return torch.log(
            torch.tril(torch.ones(size, size, device=self.generator.weight.device))
        )

    def create_pad_mask(self, x: Tensor, pad_token: int = Constants.PAD) -> Tensor:
        """Return tensor of the same shape to mask out padding tokens."""
        return x == pad_token

    def init_weights(self):
        """Initiate all weight parameters with Kaiming Uniformity."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)

    def _forward(
        self,
        x: Int[Tensor, "batch seq"],
        y: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq model"]:
        """Runs forward training pass for this seq2seq transformer training.

        Parameters:
            x (Tensor): Input sequence to train.
            y (Tensor): Output sequence to train.

        Returns:
            Tensor: output tokens by model space.
        """

        # make it causal.
        y = y[:, :-1]

        # create masks
        length = self.max_seq_length
        x_pad_mask, y_pad_mask = self.create_pad_mask(x), self.create_pad_mask(y)
        x_mask = torch.zeros((length, length), device=x.device).type(torch.bool)
        y_mask = self.generate_square_subsequent_mask(y.shape[-1]).bool()

        x_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.input_emb(x) * np.sqrt(self.dim_model)
        )
        y_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.output_emb(y) * np.sqrt(self.dim_model)
        )
        y_hat: Float[Tensor, "batch seq_len d_embed"] = super().forward(
            src=x_emb,
            tgt=y_emb,
            src_mask=x_mask,
            tgt_mask=y_mask,
            src_key_padding_mask=x_pad_mask,
            tgt_key_padding_mask=y_pad_mask,
            memory_key_padding_mask=x_pad_mask,
            tgt_is_causal=True,
        )

        return y_hat

    def forward(
        self,
        x: Int[Tensor, "batch seq"],
        y: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq generator"]:
        """Runs forward training pass for this seq2seq transformer training.

        Parameters:
            x (Tensor): Input sequence to train.
            y (Tensor): Output sequence to train.

        Returns:
            Tensor: output logits.
        """
        y_hat = self._forward(x, y)

        y_hat_tokens: Float[Tensor, "batch seq generator"] = self.generator(y_hat)

        # y_hat will return predicted tokens of y[1:], so we'll
        # copy over the original SOS token.
        sos_one_hot = torch.zeros_like(y_hat_tokens[:, 0, :])
        sos_one_hot = sos_one_hot.scatter(1, y[:, 0].unsqueeze(0).T, 1).unsqueeze(1)

        y_hat_logits = F.log_softmax(y_hat_tokens, dim=-1)
        return torch.cat([sos_one_hot, y_hat_logits], dim=1)

    def encode(
        self,
        x: Float[Tensor, "batch seq_len"],
        x_pad_mask: Optional[Bool[Tensor, "seq_len seq_len"]] = None,
    ) -> Tuple[Float[Tensor, "batch seq_len d_model"], Float[Tensor, "batch seq_len"]]:
        """Split encoder and decoder runs."""

        _, seq_len = x.shape
        x_mask = torch.zeros((seq_len, seq_len), device=x.device, dtype=torch.bool)
        x_pad_mask = x_pad_mask if x_pad_mask else self.create_pad_mask(x)
        x_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.input_emb(x) * np.sqrt(self.dim_model)
        )

        z: Float[Tensor, "batch seq_len d_model"] = self.encoder(
            x_emb, mask=x_mask, src_key_padding_mask=x_pad_mask
        )
        return z, x_pad_mask

    def decode(
        self,
        y: Float[Tensor, "batch seq_len"],
        memory: Float[Tensor, "batch seq_len d_model"],
        tgt_mask=None,
        x_pad_mask: Optional[Int[Tensor, "batch seq_len"]] = None,
    ):
        """Run decoder stage. This is needed for different decoding strategies."""
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(y.size(1))

        y_padding_mask = self.create_pad_mask(y)

        y_emb: Float[Tensor, "batch seq_len d_embed"] = self.pos_encoder(
            self.output_emb(y) * np.sqrt(self.dim_model)
        )

        return self.decoder(
            tgt=y_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=y_padding_mask,
            memory_key_padding_mask=x_pad_mask,
            tgt_is_causal=True,
        )


class PyTorchTransformerModule(NMTModule):
    def __init__(
        self,
        vocabulary_size,
        n_warmup_steps: int = 4000,
        name="PyTorchTransformer",
        sequence_length=100,
        **kwargs,
    ):
        super().__init__(
            name=name,
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            **kwargs,
        )
        self.n_warmup_steps = n_warmup_steps
        self.model = PyTorchTransformerModel(
            vocab_size=self.vocabulary_size,
            decoder_vocab_size=self.decoder_vocabulary_size,
            batch_first=True,
            max_seq_length=sequence_length,
            **kwargs,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        # lr scheduler is applied per step; not epoch.
        # so we're changing it.
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer, n_warmup_steps=self.n_warmup_steps, max_iters=3000000
        )
        return optimizer

    @torch.inference_mode()
    def beam_search(
        self,
        input_tokens: Int[Tensor, "batch seq_len"],
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
        verbose=False,
    ) -> List[Tuple[float, Tensor]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
        from the initial prompt) until either of the two stopping criteria are met:

            (1) we've generated `max_new_tokens` tokens, or
            (2) we've generated `num_returns_sequences` terminating sequences.

        To modularize this function, most of the actual complexity is in the Beams class,
        in the `generate` and `filter` methods.
        """
        if num_return_sequences > num_beams:
            raise ValueError(
                f"num_return_sequences (currently {num_return_sequences} must be <= num_beams (currently {num_beams})"
            )

        batch_size = len(input_tokens)
        # List for final beams to return (and early terminations)
        beam_scores: List[Tuple[float, Tensor]] = []
        # Keep track of all best beams after each step
        starting_log_probs = torch.zeros(input_tokens.shape[0]).to(input_tokens.device)

        memory, x_pad_mask = self.model.encode(input_tokens)
        y_hat = torch.full(
            (batch_size, 1), Constants.SOS, dtype=torch.long, device=self.device
        )

        best_beams = TransformerEncoderDecoderBeam(
            model=self.model,
            tokenizer=self.decoder_tokenizer,
            logprob_sums=starting_log_probs,
            tokens=y_hat,
            memory=memory,
            x_pad_mask=x_pad_mask,
        )

        for n in range(max_new_tokens):
            log.info(f"Debug @ n={n}")
            # Generation step
            best_beams = best_beams.generate(
                tokens_per_beam=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
            )

            # Filtering step
            best_beams, best_beams_terminated = best_beams.filter(num_beams=num_beams)
            beam_scores.extend(best_beams_terminated.logprobs_and_tensors)

            # Apply length penalty
            terminated_logprobs_and_tensors = [
                (logprob / (len(tokens) ** length_penalty), tokens)
                for logprob, tokens in best_beams_terminated.logprobs_and_tensors
            ]
            beam_scores.extend(terminated_logprobs_and_tensors)

            # Print output
            if verbose:
                best_beams.print()

            # Check stopping condition
            if len(beam_scores) >= num_return_sequences:
                break

            # Clear unnecessary tensors to free up memory
            del best_beams_terminated
            if best_beams.tokens.device == "cuda":
                torch.cuda.empty_cache()

        beam_scores.extend(best_beams.logprobs_and_tensors)
        beam_scores = beam_scores[:num_return_sequences]
        beam_scores.sort(key=lambda x: x[0], reverse=True)
        return beam_scores

    @torch.inference_mode()
    def generate(
        self,
        input_tokens: Int[Tensor, "batch seq_len"],
        max_len=50,
        start_symbol: int = Constants.SOS,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
        **kwargs,
    ) -> Int[Tensor, "batch max_len"]:
        """
        Returns a string of autoregressively generated text from the decoder.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.

        args:
            input_tokens (Int[Tensor, "batch_size seq_length"]): input sequence
                to send to the MT model.
            max_len (int): maximum length of generated sequence.
        """
        sampler = LanguageModelSampler()
        batch_size, _ = input_tokens.shape
        logits: Float[Tensor, "seq_len d_vocab"]

        # setup y output values.
        y_hat = torch.full(
            (batch_size, max_len), Constants.PAD, dtype=torch.long, device=self.device
        )
        y_hat[:, 0] = start_symbol

        # Track which sequences have finished
        y_completed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        memory, x_pad_mask = self.model.encode(input_tokens)

        for i in range(1, max_len):
            # prepare decoder output to feed into the model.
            out = self.model.decode(
                y_hat[:, :i],
                tgt_mask=None,
                memory=memory,
                x_pad_mask=x_pad_mask,
            )

            logits = torch.nn.functional.log_softmax(
                self.model.generator(out[:, -1]), dim=-1
            )

            # Sample next tokens for all unfinished sequences.
            next_tokens = torch.where(
                y_completed,
                Constants.PAD,
                sampler.sample_next_token(
                    y_hat,
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    seed=seed,
                    **kwargs,
                ),
            )
            y_hat[:, i] = next_tokens

            # damn, look at that bitwise or!
            y_completed |= next_tokens == self.decoder_tokenizer.EOS

            if y_completed.all():
                break

        return y_hat
