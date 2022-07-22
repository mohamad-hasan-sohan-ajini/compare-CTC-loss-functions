import torch
from torch import Tensor, nn

from base_model import BaseOCRModel


class NativeCTCLoss(BaseOCRModel):
    def __init__(self, alphabet: list[str], line_height: int = 32) -> None:
        super().__init__(alphabet, line_height)
        self.criterion_function = nn.CTCLoss(
            blank=0,
            reduction='sum',
            zero_infinity=True,
        )

    def criterion(
            self,
            preds: Tensor,
            x_len: Tensor,
            targets: Tensor,
            targets_len: Tensor,
    ):
        # prepare target, convert from flatten to (N, S) shape
        max_target_len = targets_len.max()
        batch_size = targets_len.size(0)
        targets_with_pad = torch.zeros(batch_size, max_target_len).long()
        start = 0
        for index, target_len in enumerate(targets_len):
            end = start + target_len
            targets_with_pad[index, :target_len] = targets[start:end]
            start = end
        # calculate loss
        loss = self.criterion_function(
            preds.log_softmax(-1),
            targets_with_pad.detach().long(),
            x_len.long(),
            targets_len.long(),
        )
        return loss


class CUDNNCTCLoss(BaseOCRModel):
    def __init__(self, alphabet: list[str], line_height: int = 32) -> None:
        super().__init__(alphabet, line_height)
        self.criterion_function = nn.CTCLoss(
            blank=0,
            reduction='sum',
            zero_infinity=True,
        )

    def criterion(
            self,
            preds: Tensor,
            x_len: Tensor,
            targets: Tensor,
            targets_len: Tensor,
    ):
        # prepare x_len such that all the inputs have fixed size
        x_len_fix = torch.IntTensor([preds.size(0) for i in range(preds.size(1))])
        loss = self.criterion_function(
            preds.log_softmax(-1),
            targets.int().cpu(),
            x_len_fix.int().cpu(),
            targets_len.int().cpu(),
        )
        return loss


class WarpCTCLoss(BaseOCRModel):
    def __init__(self, alphabet: list[str], line_height: int = 32) -> None:
        super().__init__(alphabet, line_height)

    def criterion(
            self,
            preds: Tensor,
            x_len: Tensor,
            targets: Tensor,
            targets_len: Tensor,
    ):
        pass
