from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data_module import LineDataModule
from model import (NativeCTCLoss, CUDNNCTCLoss)

datamodule = LineDataModule(
    train_text_path='resources/text/Pride and Prejudice.txt',
    val_text_path='resources/text/The Call of the Wild 32lines.txt',
    test_text_path='resources/text/The Great Gatsby 32lines.txt',
    font_path_list=[
        'resources/fonts/arial/arial.ttf',
        'resources/fonts/Calibri/Calibri.ttf',
    ]
)
datamodule.setup()
model = NativeCTCLoss(
    datamodule.train_dataset.alphabet_list,
    datamodule.train_dataset.line_height,
)
checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor='val_cer',
    mode='min',
    save_last=True,
    every_n_train_steps=100,
)
lr_callback = LearningRateMonitor('step')
trainer = Trainer(
    gpus=1,
    max_epochs=3,
    progress_bar_refresh_rate=10,
    callbacks=[checkpoint_callback, lr_callback],
    gradient_clip_val=100,
    # precision=16,
)
trainer.fit(model, datamodule)
