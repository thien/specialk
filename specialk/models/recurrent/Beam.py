from typing import List, Tuple
import torch
from torch import Tensor

from specialk import Constants


class Beam:
    """
    Manages the internals of the beam search process.

    Takes care of beams, back pointers, and scores.
    """

    def __init__(self, size: int, device: torch.device):
        """
        Initialize the beam.

        Args:
            size (int): The beam size.
            device (torch.device): The device to use for tensor operations.
        """
        self.size = size
        self.done = False
        self.device = device

        # The score for each translation on the beam.
        self.scores = torch.zeros(size, device=device)

        # The backpointers at each time-step.
        self.prevKs: List[Tensor] = []

        # The outputs at each time-step.
        self.nextYs: List[Tensor] = [
            torch.full((size,), Constants.PAD, dtype=torch.long, device=device)
        ]
        self.nextYs[0][0] = Constants.SOS

        # The attentions (matrix) for each time.
        self.attn: List[Tensor] = []

    def get_current_state(self) -> Tensor:
        """Get the outputs for the current timestep."""
        return self.nextYs[-1]

    def get_current_origin(self) -> Tensor:
        """Get the backpointers for the current timestep."""
        return self.prevKs[-1]

    def advance(self, word_lk: Tensor, attn_out: Tensor) -> bool:
        """
        Compute and update the beam search.

        Args:
            word_lk (Tensor): Probs of advancing from the last step (K x words)
            attn_out (Tensor): Attention at the last step

        Returns:
            bool: True if beam search is complete.
        """
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(best_scores_id - prev_k * num_words)
        self.attn.append(attn_out.index_select(0, prev_k))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == Constants.EOS:
            self.done = True

        return self.done

    def sort_best(self) -> Tuple[Tensor, Tensor]:
        """Sort the beam by score."""
        return torch.sort(self.scores, 0, True)

    def get_best(self) -> Tuple[Tensor, Tensor]:
        """Get the score of the best in the beam."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k: int) -> Tuple[List[int], Tensor]:
        """
        Walk back to construct the full hypothesis.

        Args:
            k (int): The position in the beam to construct.

        Returns:
            Tuple[List[int], Tensor]: The hypothesis and the attention at each time step.
        """
        hyp: List[int] = []
        attn: List[Tensor] = []

        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k].item())
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k].item()

        return hyp[::-1], torch.stack(attn[::-1])
