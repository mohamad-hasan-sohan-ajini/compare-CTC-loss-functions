'''Base Model'''

from itertools import chain

import torch
from torch import Tensor, nn
from torchmetrics import CharErrorRate, WordErrorRate
from pytorch_lightning import LightningModule


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple,
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class BGRU(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        """forward

        :param x: Input tensor of shape (T, N, H)
        :type x: Tensor
        :return: Time smoothed tensor
        :rtype: Tensor
        """
        x, _ = self.gru(x)
        timesteps, batch_size, _ = x.size()
        x = x.view(timesteps, batch_size, 2, -1).sum(dim=2)
        return x


class GreedyCTCDecoder(nn.Module):
    def __init__(self, labels, blank_token, space_token):
        super().__init__()
        self.labels = labels
        self.blank_index = labels.index(blank_token)
        self.space_token = space_token

    def forward(self, emission: Tensor) -> list[str]:
        # emission: logit tensors. Shape (T, |Alphabet|).
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_index]
        joined = ''.join([self.labels[i] for i in indices])
        return joined.replace(self.space_token, ' ').strip()


class BaseOCRModel(LightningModule):
    def __init__(
            self,
            alphabet: list[str],
            line_height: int = 32,
            max_channels: int = 512,
    ) -> None:
        super().__init__()
        self.alphabet = alphabet
        # evaluation tools
        self.line_height = line_height
        self.max_channels = max_channels
        self.ctc_decoder = GreedyCTCDecoder(
            labels=alphabet,
            blank_token=alphabet[0],
            space_token=alphabet[1],
        )
        self.cer_metric = CharErrorRate()
        self.wer_metric = WordErrorRate()
        # backbone model
        self.conv = nn.Sequential()
        in_channels = 1
        out_channels = 32
        module_name_counter = 0
        while line_height > 1:
            self.conv.add_module(
                f'layer{module_name_counter}_conv0',
                ConvBlock(in_channels, out_channels, 3),
            )
            self.conv.add_module(
                f'layer{module_name_counter}_conv1',
                ConvBlock(out_channels, out_channels, 3),
            )
            self.conv.add_module(
                f'layer{module_name_counter}_maxpool',
                nn.AvgPool2d((2, 1), (2, 1)),
            )
            line_height = line_height // 2
            module_name_counter += 1
            in_channels = out_channels
            out_channels *= 2
            out_channels = min(out_channels, self.max_channels)
        self.rnn = BGRU(in_channels, in_channels)
        self.linear = nn.Linear(in_channels, len(self.alphabet), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        :param x: Input images. Shape: (N, C, H, T)
        :type x: Tensor
        :return: Backbone network raw output. Shape (T, N, Alphabet)
        :rtype: Tensor
        """
        x = 1 - x.mean(dim=1, keepdim=True)
        x = self.conv(x)
        # squeeze height and reform data to (T, N, Alphabet) shape
        x = x.squeeze(2).permute(2, 0, 1)
        x = self.rnn(x)
        x = self.linear(x)
        return x

    def criterion(
            self,
            pred: Tensor,
            x_len: Tensor,
            targets: Tensor,
            targets_len: Tensor,
    ) -> Tensor:
        """Calculate CTC loss"""
        raise NotImplemented(
            'Successor class must implement this based on CTC loss '
            'considerations.'
        )

    def training_step(self, batch, batch_idx) -> dict[str, Tensor]:
        x, x_len, targets, targets_len = batch
        pred = self(x)
        loss = self.criterion(pred, x_len, targets, targets_len)
        self.log('train_loss', loss.detach().cpu().item())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx) -> dict[str, Tensor]:
        x, x_len, targets, targets_len = batch
        pred = self(x)
        loss = self.criterion(pred, x_len, targets, targets_len)
        cer, wer = self._calculate_metrics(
            pred.detach().cpu(),
            targets.cpu(),
            x_len.cpu(),
            targets_len.cpu(),
        )
        self.log('val_loss', loss.detach().cpu().item())
        self.log('val_cer', cer, on_epoch=True)
        self.log('val_wer', wer, on_epoch=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx) -> dict[str, Tensor]:
        x, x_len, targets, targets_len = batch
        pred = self(x)
        loss = self.criterion(pred, x_len, targets, targets_len)
        cer, wer = self._calculate_metrics(
            pred.detach(),
            targets.cpu(),
            x_len.cpu(),
            targets_len.cpu(),
        )
        self.log('test_loss', loss.detach().cpu().item())
        self.log('test_cer', cer)
        self.log('test_wer', wer)
        return {'loss': loss}

    def _decode_preds(self, preds: Tensor, preds_len: Tensor) -> list[str]:
        """Decode network predictions

        :param preds: Probabilities of each character. Shape
            (T, N, Alphabet)
        :type preds: Tensor
        :param preds_len: Length of inputs image regards each sample.
            Shape (N)
        :type preds_len: Tensor
        :return: Decoded strings
        :rtype: list[str]
        """
        hyps = []
        for pred, pred_len in zip(preds.transpose(0, 1), preds_len):
            try:
                hyps.append(self.ctc_decoder(pred[:pred_len]))
            except IndexError:
                hyps.append('')
        return hyps

    def _decode_targets(
            self,
            targets: Tensor,
            targets_len: Tensor,
    ) -> list[str]:
        """Decode targets

        :param targets: Flattened targets. Shape (None)
        :type targets: Tensor
        :param targets_len: Length of targets regards each sample.
            Shape (N)
        :type targets_len: Tensor
        :return: Decoded strings
        :rtype: list[str]
        """
        targets_len_cs = targets_len.cumsum(0)
        targets_len_cs_0 = torch.cat([torch.IntTensor([0]), targets_len_cs])
        refs = [
            ''.join(
                [
                    self.alphabet[token_index]
                    for token_index in targets[start:end]
                ]
            )
            for start, end in zip(targets_len_cs_0, targets_len_cs)
        ]
        return refs

    def _calculate_metrics(
            self,
            preds: Tensor,
            targets: Tensor,
            preds_len: Tensor,
            targets_len: Tensor,
    ) -> tuple[float, float]:
        """Calculate CER and WER values

        :param preds: Probabilities of each character. Shape
            (T, N, Alphabet)
        :type preds: Tensor
        :param targets: Flattened targets. Shape (None)
        :type targets: Tensor
        :param preds_len: Length of inputs image regards each sample.
            Shape (N)
        :type preds_len: Tensor
        :param targets_len: Length of targets regards each sample.
            Shape (N)
        :type targets_len: Tensor
        :return: The value of CER and WER metrics
        :rtype: tuple[float, float]
        """
        hyps = self._decode_preds(preds, preds_len)
        refs = self._decode_targets(targets, targets_len)
        return self.cer_metric(hyps, refs), self.wer_metric(hyps, refs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(
                self.conv.parameters(),
                self.rnn.parameters(),
                self.linear.parameters(),
            ),
            lr=2e-4,
        )
        return {'optimizer': optimizer}
