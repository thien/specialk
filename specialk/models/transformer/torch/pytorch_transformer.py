"""
PyTorch native implementation of the Transformer.

This is intentional to take advantage of native
implementations (as reasonably as possible), so a lot of this
will be calling the relevant classes and modified afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from galore_torch import GaLoreAdamW
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import LayerNorm

import specialk.core.constants as Constants
from specialk.core.utils import log
from specialk.models.generators.beam import BeamBatch, EncoderDecoderBeam
from specialk.models.generators.sampling import LanguageModelSampler
from specialk.models.mt_model import NMTModule
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

    def new(
        self,
        logprob_sums: Float[Tensor, "batch"],
        tokens: Int[Tensor, "batch seq"],
        memory: Tensor,
        x_pad_mask: Tensor,
    ) -> TransformerEncoderDecoderBeam:
        """Creates a new EncoderDecoderBeam object with the
        same model, tokenizer, and encoder_output. Includes an x_pad_mask."""
        return TransformerEncoderDecoderBeam(
            model=self.model,
            tokenizer=self.tokenizer,
            logprob_sums=logprob_sums,
            tokens=tokens,
            memory=memory,
            x_pad_mask=x_pad_mask,
        )

    @torch.inference_mode()
    def get_logits(self) -> Float[Tensor, "batch vocab"]:
        logits = self.model.decode(
            self.tokens,
            tgt_mask=None,
            memory=self.memory,
            x_pad_mask=self.x_pad_mask,
        )
        return torch.nn.functional.log_softmax(
            self.model.generator(logits[:, -1]), dim=-1
        )

    def __getitem__(self, idx) -> TransformerEncoderDecoderBeam:
        return self.new(
            self.logprob_sums[idx],
            self.tokens[idx],
            self.memory[idx],
            self.x_pad_mask[idx],
        )

    def generate(
        self,
        tokens_per_beam: int,
        no_repeat_ngram_size: Optional[int] = None,
        log_logits: Optional[Float[Tensor, "batch vocab"]] = None,
    ) -> TransformerEncoderDecoderBeam:
        if log_logits is None:
            log_logits = self.get_logits()

        log_probs, tok_idx = self.get_topk_non_repeating(
            log_logits, no_repeat_ngram_size, k=tokens_per_beam
        )

        new_logprob_sums = self._calculate_new_logprob_sums(log_probs, tokens_per_beam)
        new_tokens = self._generate_new_tokens(tok_idx, tokens_per_beam)
        new_encoder_outputs = self._repeat_memory(k=tokens_per_beam)
        new_x_pad_mask = einops.repeat(
            self.x_pad_mask, "b s -> (b beam) s", beam=tokens_per_beam
        )
        return self.new(
            new_logprob_sums, new_tokens, new_encoder_outputs, new_x_pad_mask
        )


@dataclass
class TransformerBeamBatch(BeamBatch):
    beams: List[TransformerEncoderDecoderBeam]

    @torch.inference_mode()
    def get_logits(self) -> List[Float[torch.Tensor, "beam*beam vocab"]]:
        # Aggregate inputs from all beams
        all_tokens = torch.cat([beam.tokens for beam in self.beams])
        all_memory = torch.cat([beam.memory for beam in self.beams])
        all_x_pad_mask = torch.cat([beam.x_pad_mask for beam in self.beams])
        decode = self.beams[0].model.decode
        generator = self.beams[0].model.generator

        # Perform the logits computation in one go
        logits: Float[Tensor, "batch*beam vocab"] = decode(
            all_tokens,
            tgt_mask=None,
            memory=all_memory,
            x_pad_mask=all_x_pad_mask,
        )[:, -1]
        log_probs: Float[Tensor, "batch*beam vocab"]
        log_probs = torch.nn.functional.log_softmax(generator(logits), dim=-1)

        # Disperse the results back to individual beams
        return list(log_probs.split([beam.num_beams for beam in self.beams]))


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
        # galore_params are essentially all attn and ff layers in the model.
        galore_params, non_galore_params = [], []

        # Iterate through named parameters
        for name, param in self.model.named_parameters():
            if "linear" in name or "self_attn" in name:
                galore_params.append(param)
            else:
                non_galore_params.append(param)

        param_groups = [
            {"params": non_galore_params},
            {
                "params": galore_params,
                "rank": 128,
                "update_proj_gap": 200,
                "scale": 0.25,
                "proj_type": "std",
            },
        ]
        optimizer = GaLoreAdamW(
            param_groups, lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-9
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
    ) -> Tuple[
        Float[Tensor, "batch n_return_seq"], Int[Tensor, "batch n_return_seq seq_len"]
    ]:
        """
        Implements a beam search.
        """
        batch_size = input_tokens.size(0)
        device = input_tokens.device

        if num_return_sequences > num_beams:
            raise ValueError(
                f"num_return_sequences (currently {num_return_sequences} "
                f"must be <= num_beams (currently {num_beams})"
            )

        # encode input sequences
        memory, x_pad_mask = self.model.encode(input_tokens)

        # initialize beams
        beams = []
        for i in range(batch_size):
            y_hat = torch.full(
                (num_beams, 1), Constants.SOS, dtype=torch.long, device=device
            )
            logprob_sums = torch.zeros(num_beams, device=device)

            # expands memory and x_pad_mask multiple times to the length of beams
            # (for batch operation).
            item_memory = memory[i].unsqueeze(0).expand(num_beams, -1, -1)
            item_x_pad_mask = x_pad_mask[i].unsqueeze(0).expand(num_beams, -1)

            beams.append(
                TransformerEncoderDecoderBeam(
                    model=self.model,
                    tokenizer=self.decoder_tokenizer,
                    logprob_sums=logprob_sums,
                    tokens=y_hat,
                    memory=item_memory,
                    x_pad_mask=item_x_pad_mask,
                )
            )
        batch_of_beams = TransformerBeamBatch.create(beams)

        # List for final beams to return (and early terminations)
        finished_beams: List[List[Tuple[float, Tensor]]]
        finished_beams = [[] for _ in range(batch_size)]

        for n in range(max_new_tokens):
            # Generation step
            batch_of_beams = batch_of_beams.generate(num_beams, no_repeat_ngram_size)
            batch_of_beams, terminated_beams = batch_of_beams.filter(num_beams)

            # Process terminated beams
            for i, beam in enumerate(terminated_beams.beams):
                if beam.logprob_sums.size(0) > 0:
                    scores = beam.logprob_sums / (beam.tokens.size(1) ** length_penalty)
                    finished_beams[i].extend(list(zip(scores.tolist(), beam.tokens)))

            # Check if all beams are finished
            if all(len(beams) >= num_return_sequences for beams in finished_beams):
                break

        # Print output
        if verbose:
            batch_of_beams.print(title=f"Best completions @ n={n+1}")

        # Add remaining beams
        for i, beam in enumerate(batch_of_beams.beams):
            scores = beam.logprob_sums / (beam.tokens.size(1) ** length_penalty)
            finished_beams[i].extend(list(zip(scores.tolist(), beam.tokens)))

        # Sort and truncate results
        for i in range(batch_size):
            finished_beams[i].sort(key=lambda x: x[0], reverse=True)
            finished_beams[i] = finished_beams[i][:num_return_sequences]

        # Prepare the output tensors
        max_seq_len = max(max(len(seq) for _, seq in beams) for beams in finished_beams)
        scores_tensor = torch.zeros((batch_size, num_return_sequences), device=device)
        tokens_tensor = torch.full(
            (batch_size, num_return_sequences, max_seq_len),
            self.tokenizer.PAD,
            dtype=torch.long,
            device=device,
        )

        for i, batch_beams in enumerate(finished_beams):
            # Sort and truncate results
            batch_beams.sort(key=lambda x: x[0], reverse=True)
            batch_beams = batch_beams[:num_return_sequences]

            for j, (score, seq) in enumerate(batch_beams):
                scores_tensor[i, j] = score
                tokens_tensor[i, j, : len(seq)] = seq

        log.info("scores", sc=scores_tensor.shape, tok=tokens_tensor.shape)
        return scores_tensor, tokens_tensor

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
