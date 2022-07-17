from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset import LineDataset, collate_function


class LineDataModule(LightningDataModule):
    def __init__(
            self,
            train_text_path: str,
            val_text_path: str,
            test_text_path: str,
            font_path_list: list[str],
            line_height: int = 32,
            batch_size: int = 32,
            num_workers: int = 8,
    ) -> None:
        self.train_text_path = train_text_path
        self.val_text_path = val_text_path
        self.test_text_path = test_text_path
        self.font_path_list = font_path_list
        self.line_height = line_height
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = LineDataset(
            self.train_text_path,
            self.font_path_list,
        )
        self.val_dataset = LineDataset(
            self.val_text_path,
            self.font_path_list,
        )
        self.test_dataset = LineDataset(
            self.test_text_path,
            self.font_path_list,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_function,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_function,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_function,
        )


if __name__ == '__main__':
    dm = LineDataModule(
        train_text_path='resources/text/Pride and Prejudice.txt',
        val_text_path='resources/text/The Call of the Wild.txt',
        test_text_path='resources/text/The Great Gatsby.txt',
        font_path_list=[
            'resources/fonts/arial/arial.ttf',
            'resources/fonts/Calibri/Calibri.ttf',
        ]
    )
    dm.setup()
    dl = dm.train_dataloader()
    for x, x_len, target, target_len in dl:
        break
