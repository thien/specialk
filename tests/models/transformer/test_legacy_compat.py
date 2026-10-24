from argparse import Namespace

import torch

import specialk.core.constants as Constants
from specialk.models.transformer.legacy.Models import Transformer
from specialk.models.transformer.torch.legacy_compat import LegacyTransformerModel
from specialk.models.transformer.torch.modules import cross_kv


def _legacy_checkpoint(tmp_path):
    settings = Namespace(
        src_vocab_size=20,
        tgt_vocab_size=20,
        max_token_seq_len=10,
        proj_share_weight=False,
        embs_share_weight=False,
        d_k=4,
        d_v=4,
        d_model=8,
        d_word_vec=8,
        d_inner_hid=16,
        layers=1,
        n_head=2,
        dropout=0.0,
    )
    legacy = Transformer(
        n_src_vocab=settings.src_vocab_size,
        n_tgt_vocab=settings.tgt_vocab_size,
        len_max_seq=settings.max_token_seq_len,
        tgt_emb_prj_weight_sharing=settings.proj_share_weight,
        emb_src_tgt_weight_sharing=settings.embs_share_weight,
        d_k=settings.d_k,
        d_v=settings.d_v,
        d_model=settings.d_model,
        d_word_vec=settings.d_word_vec,
        d_inner=settings.d_inner_hid,
        n_layers=settings.layers,
        n_head=settings.n_head,
        dropout=settings.dropout,
    )
    legacy.eval()
    encoder_path = tmp_path / "encoder.chkpt"
    decoder_path = tmp_path / "decoder.chkpt"
    torch.save(
        {
            "settings": settings,
            "epoch": 1,
            "model": {f"module.{k}": v for k, v in legacy.encoder.state_dict().items()},
        },
        encoder_path,
    )
    torch.save(
        {
            "settings": settings,
            "epoch": 1,
            "model": {f"module.{k}": v for k, v in legacy.decoder.state_dict().items()},
            "generator": legacy.generator.state_dict(),
        },
        decoder_path,
    )
    return legacy, encoder_path, decoder_path


def test_legacy_checkpoint_matches_2019_forward(tmp_path):
    legacy, encoder_path, decoder_path = _legacy_checkpoint(tmp_path)
    modern, _ = LegacyTransformerModel.from_checkpoints(encoder_path, decoder_path)

    src = torch.tensor([[Constants.SOS, 5, 6, Constants.EOS, Constants.PAD]])
    tgt = torch.tensor([[Constants.SOS, 7, 8, Constants.EOS]])
    src_pos = torch.tensor([[1, 2, 3, 4, 0]])
    tgt_pos = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        expected = legacy(src, src_pos, tgt, tgt_pos)
        actual = modern.forward_logits(src, tgt)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        expected_encoded = legacy.encoder(src, src_pos)[0]
        actual_encoded, pad_mask = modern.encode(src)
        torch.testing.assert_close(actual_encoded, expected_encoded, atol=1e-5, rtol=1e-5)

        memory_kv = cross_kv(modern.decoder, actual_encoded)
        expected_step = legacy.decoder(tgt[:, :1], tgt_pos[:, :1], src, expected_encoded)[0]
        actual_step, _ = modern.decode_step(tgt[:, :1], memory_kv, pad_mask)
        torch.testing.assert_close(actual_step, expected_step, atol=1e-5, rtol=1e-5)
