import torch
from torch import Tensor, nn

from base_model import BaseOCRModel


class NativeCTCLoss(BaseOCRModel):
    def __init__(self, alphabet: list[str], line_height: int = 32) -> None:
        super().__init__(alphabet, line_height)

    def criterion(self, pred: Tensor, x_len: Tensor, targets: Tensor, targets_len: Tensor):
        # prepare target, convert from flatten to (N, S) shape
        max_targets_len = targets_len.max()
        batch_size = targets_len.size(0)
        targets_padded = torch.zeros(batch_size, max_targets_len).long()


class CUDNNCTCLoss(BaseOCRModel):
    def __init__(self, alphabet: list[str], line_height: int = 32) -> None:
        super().__init__(alphabet, line_height)

    def criterion(self, pred: Tensor, x_len: Tensor, targets: Tensor, targets_len: Tensor):
        pass


class WarpCTCLoss(BaseOCRModel):
    def __init__(self, alphabet: list[str], line_height: int = 32) -> None:
        super().__init__(alphabet, line_height)

    def criterion(self, pred: Tensor, x_len: Tensor, targets: Tensor, targets_len: Tensor):
        pass
