# import onmt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import Beam, Constants, Dataset, Models, Models_decoder, modules


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(
            opt.decoder_model, map_location=lambda storage, loc: storage
        )
        encoder_check = torch.load(
            opt.encoder_model, map_location=lambda storage, loc: storage
        )

        self.src_dict = encoder_check["dicts"]["src"]
        self.tgt_dict = checkpoint["dicts"]["tgt"]
        enc_opt = encoder_check["opt"]

        encoder = Models.Encoder(enc_opt, self.src_dict)
        encoder.load_state_dict(encoder_check["encoder"])
        decoder = Models_decoder.Decoder(enc_opt, self.tgt_dict)
        decoder.load_state_dict(checkpoint["decoder"])
        model = Models.NMTModel(encoder, decoder)

        generator = nn.Sequential(
            nn.Linear(enc_opt.rnn_size, self.tgt_dict.size()), nn.LogSoftmax()
        )

        generator.load_state_dict(checkpoint["generator"])

        if opt.cuda:
            encoder.cuda()
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.encoder = encoder
        self.model = model
        self.model.eval()

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b, Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [
                self.tgt_dict.convertToIdx(
                    b, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD
                )
                for b in goldBatch
            ]

        return Dataset(
            srcData, tgtData, self.opt.batch_size, self.opt.cuda, volatile=True
        )

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex.item()]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.encoder(srcBatch)
        srcBatch = srcBatch[0]  # drop the lengths needed for encoder

        rnnSize = context.size(2)
        encStates = (
            self.model._fix_enc_hidden(encStates[0]),
            self.model._fix_enc_hidden(encStates[1]),
        )

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(Constants.PAD).t()

        def applyContextMask(m):
            if isinstance(m, modules.GlobalAttention):
                m.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()
        if tgtBatch is not None:
            decStates = encStates
            decOut = self.model.make_init_decoder_output(context)
            self.model.decoder.apply(applyContextMask)
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

        beam = [Beam(beamSize, self.opt.cuda) for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(context)

        padMask = (
            srcBatch.data.eq(Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)
        )

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):
            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            input = (
                torch.stack([b.getCurrentState() for b in beam if not b.done])
                .t()
                .contiguous()
                .view(1, -1)
            )

            decOut, decStates, attn = self.model.decoder(
                Variable(input, volatile=True), decStates, context, decOut
            )
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

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
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin())
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(
                    view.index_select(1, activeIdx).view(*newSize), volatile=True
                )

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]

        return allHyp, allScores, allAttn, goldScores

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, indices = dataset[0]

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(src, tgt)
        pred, predScore, attn, goldScore = list(
            zip(
                *sorted(
                    zip(pred, predScore, attn, goldScore, indices), key=lambda x: x[-1]
                )
            )
        )[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [
                    self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                    for n in range(self.opt.n_best)
                ]
            )

        return predBatch, predScore, goldScore
