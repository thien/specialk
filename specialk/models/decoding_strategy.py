import torch


class DecoderStrategy:
    def __init__(self, model):
        self.model = model


class GreedyDecoder(DecoderStrategy):
    def __init__(self, model):
        super().__init__(model)

    def translate(self, batch) -> torch.Tensor:
        self.model(batch)
