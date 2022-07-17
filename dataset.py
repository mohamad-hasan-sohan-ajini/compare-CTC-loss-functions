import random
import string

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import PILToTensor


class LineDataset(Dataset):
    alphabet_list = (
        ['—', ' ']
        + list(string.ascii_letters)
        + list(string.digits)
        + list(string.punctuation)
    )
    line_height = 32
    x_margin = 16
    y_margin = 16

    def __init__(
            self,
            text_path: str,
            font_path_list: list[str],
    ) -> None:
        self.alphabet_dict = {
            char: index
            for index, char in enumerate(self.alphabet_list)
        }
        # load text book + preprocessing
        with open(text_path) as f:
            text = f.read().splitlines()
        self.text = self._preprocess(text)
        # load fonts
        self.fonts = [
            ImageFont.truetype(font_path, 32)
            for font_path in font_path_list
        ]

    def _preprocess(self, text: list[str]) -> list[str]:
        # remove empty lines and strip whitespaces
        text = [line.strip() for line in text if line]
        # remove lines contain characters not in alphabet_list
        alphabet_set = set(self.alphabet_list)
        text = [
            line
            for line in text
            if all([ch in alphabet_set for ch in line])
        ]
        # sort by length
        text = sorted(text, key=lambda x: len(x))
        return text

    def _get_image_of_line(self, line: str) -> tuple[torch.Tensor, int]:
        # calculate canvas size
        font = random.choice(self.fonts)
        (left, _, right, _) = font.getbbox(line)
        width = right - left + self.x_margin
        height = font.getmetrics()[0] + self.y_margin
        # create canvas and write line
        canvas = Image.new('RGB', (width, height), "#FFFFFF")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (self.x_margin // 2, self.y_margin // 2),
            line,
            font=font,
            fill='#000000',
        )
        # rescale to line height
        scale = self.line_height / height
        canvas = canvas.resize((int(width * scale), self.line_height))
        image_tensor = PILToTensor()(canvas)
        return image_tensor / 255, image_tensor.size(2)

    def _get_target_of_line(self, line: str) -> tuple[torch.Tensor, int]:
        target = torch.IntTensor([self.alphabet_dict[ch] for ch in line])
        return target, target.size(0)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        line = self.text[index]
        x, x_len = self._get_image_of_line(line)
        target, target_len = self._get_target_of_line(line)
        return x, x_len, target, target_len


def cudnn_compatible_collate_function(batch):
    x_batch, x_len_batch, target_batch, target_len_batch = zip(*batch)
    batch_size = len(x_batch)
    time_steps = max(x_len_batch)
    num_freqs = x_batch[0].size(1)
    x_result = torch.zeros((batch_size, 3, num_freqs, time_steps))
    for index, x in enumerate(x_batch):
        x_time_steps = x.size(2)
        x_result[index, :, :, :x_time_steps] = x
    x_len_result_fix = torch.IntTensor([time_steps] * batch_size)
    x_len_result_real = torch.IntTensor(x_len_batch)
    target_result = torch.cat(target_batch)
    target_len_result = torch.IntTensor(target_len_batch).int()
    result = (
        x_result,
        x_len_result_fix,
        x_len_result_real,
        target_result,
        target_len_result,
    )
    return result


if __name__ == '__main__':
    ds = LineDataset(
        'resources/text/Pride and Prejudice.txt',
        [
            'resources/fonts/arial/arial.ttf',
            'resources/fonts/Calibri/Calibri.ttf',
        ],
    )
    dl = DataLoader(
        ds,
        batch_size=8,
        num_workers=4,
        collate_fn=cudnn_compatible_collate_function,
    )
    for x, x_len_fix, x_len_real, target, target_len in dl:
        break
