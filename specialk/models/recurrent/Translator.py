import torch
import torch.nn as nn
from torch.autograd import Variable

from specialk import Constants
from specialk.models.recurrent import Models
from specialk.models.recurrent.Beam import Beam
from specialk.models.recurrent.GlobalAttention import GlobalAttention


class Translator(object):
    def __init__(self, opt, new=True):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")
        if new:
            # self.tt = torch.cuda if opt.cuda else torch

            checkpoint = torch.load(opt.model)

            model_opt = checkpoint["opt"]
            self.src_dict = checkpoint["dicts"]["src"]
            self.tgt_dict = checkpoint["dicts"]["tgt"]

            encoder = Models.Encoder(model_opt, self.src_dict)
            decoder = Models.Decoder(model_opt, self.tgt_dict)
            model = Models.NMTModel(encoder, decoder)

            # @ change here.
            generator = nn.Sequential(
                nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
                nn.LogSoftmax(dim=1),
            )

            if "model" in checkpoint.keys():
                model.load_state_dict(checkpoint["model"])

            generator.load_state_dict(checkpoint["generator"])

            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()

            model.generator = generator

            self.model = model
            self.model.eval()

    def translate_batch(self, src_seq, src_pos):
        """Translation work in one batch"""

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """Indicate the position of an instance in a tensor."""
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(
            beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm
        ):
            """Collect tensor parts associated to active instances."""

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
            src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list
        ):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(
                src_seq, active_inst_idx, n_prev_active_inst, n_bm
            )
            active_src_enc = collect_active_part(
                src_enc, active_inst_idx, n_prev_active_inst, n_bm
            )
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
            inst_dec_beams,
            len_dec_seq,
            src_seq,
            enc_output,
            inst_idx_to_position_map,
            n_bm,
        ):
            """Decode and update beam status, and then return active beam idx"""

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(
                    1, len_dec_seq + 1, dtype=torch.long, device=self.device
                )
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(
                    n_active_inst * n_bm, 1
                )
                return dec_partial_pos

            def collect_active_inst_idx_list(
                inst_beams, word_prob, inst_idx_to_position_map
            ):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(
                        word_prob[inst_position]
                    )
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            # word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            out, _, _ = self.decoder(dec_seq, src_enc, context, enc_output)
            word_prob = out.view(n_active_inst, n_bm, -1)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [
                    inst_dec_beams[inst_idx].get_hypothesis(i)
                    for i in tail_idxs[:n_best]
                ]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # -- Encode
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            # src_enc, *_ = self.model.encoder(src_seq, src_pos)
            src_enc, context = self.model.encoder((src_seq, src_pos))
            rnnSize = context.size(2)
            src_enc = (
                self.model._fix_enc_hidden(src_enc[0]),
                self.model._fix_enc_hidden(src_enc[1]),
            )

            # -- Repeat data for beam search
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            # -- Decode
            for len_dec_seq in range(1, self.max_token_seq_len + 1):
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_seq,
                    src_enc,
                    inst_idx_to_position_map,
                    n_bm,
                )

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list
                )

        batch_hyp, batch_scores = collect_hypothesis_and_scores(
            inst_dec_beams, self.opt.n_best
        )

        return batch_hyp, batch_scores

    def translateBatch(self, srcBatch, tgtBatch=None):
        src_seq, src_len = srcBatch
        src_seq = src_seq.transpose(0, 1)
        srcBatch = (src_seq, src_len)
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        context, encStates = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0]  # drop the lengths needed for encoder

        rnnSize = context.size(2)
        encStates = (
            self.model._fix_enc_hidden(encStates[0]),
            self.model._fix_enc_hidden(encStates[1]),
        )

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(Constants.PAD).t()

        def apply_context_mask(m):
            if isinstance(m, GlobalAttention):
                m.mask = padMask

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()

        # This section is most likely not to be called.
        if tgtBatch is not None:
            decStates = encStates
            decOut = self.model.make_init_decoder_output(context)
            self.model.decoder.apply(apply_context_mask)
            initOutput = self.model.make_init_decoder_output(context)

            decOut, decStates, attn = self.model.decoder(
                tgtBatch[:-1], decStates, context, initOutput
            )
            for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(Constants.PAD), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = Variable(context.data.repeat(1, beamSize, 1))
        decStates = (
            Variable(encStates[0].data.repeat(1, beamSize, 1)),
            Variable(encStates[1].data.repeat(1, beamSize, 1)),
        )

        beam = [Beam(beamSize, context.device) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = (
            srcBatch.data.eq(Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)
        )

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.max_token_seq_len):
            self.model.decoder.apply(apply_context_mask)

            # Prepare decoder input.
            input = (
                torch.stack([b.get_current_state() for b in beam if not b.done])
                .t()
                .contiguous()
                .view(1, -1)
            )

            with torch.no_grad():
                input = Variable(input)
            decOut, decStates, attn = self.model.decoder(
                input, decStates, context, decOut, False
            )
            # decOut: 1 x (beam*batch) x numWords
            out = self.model.decoder.generator(decOut)
            out = out.transpose(0, 1)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, beamSize, remainingSents, decState.size(2)
                    )[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].get_current_origin())
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = torch.LongTensor([batchIdx[k] for k in active]).to(self.device)
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                with torch.no_grad():
                    return Variable(view.index_select(1, activeIdx).view(*newSize))

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]

        return allHyp, allScores, allAttn, goldScores

    # def translate(self, srcBatch, goldBatch):
    #     #  (1) convert words to indexes
    #     dataset = self.buildData(srcBatch, goldBatch)
    #     src, tgt, indices = dataset[0]

    #     #  (2) translate
    #     pred, predScore, attn, goldScore = self.translateBatch(src, tgt)
    #     pred, predScore, attn, goldScore = list(zip(*sorted(zip(pred, predScore, attn, goldScore, indices), key=lambda x: x[-1])))[:-1]

    #     #  (3) convert indexes to words
    #     predBatch = []
    #     for b in range(src[0].size(1)):
    #         predBatch.append(
    #             [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
    #                     for n in range(self.opt.n_best)]
    #         )

    #     return predBatch, predScore, goldScore
