import matplotlib.pyplot as plt
import torch

from data_module import LineDataModule
from model import BaseOCRModel

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
for x, x_len, targets, targets_len in dl:
    break

model = BaseOCRModel(
    dm.train_dataset.alphabet_list,
    dm.train_dataset.line_height,
)
model = model.load_from_checkpoint(
    'lightning_logs/version_18/checkpoints/epoch=93-step=1001200.ckpt',
    alphabet=dm.train_dataset.alphabet_list,
    line_height=dm.train_dataset.line_height,
)

with torch.inference_mode():
    preds = model(x)

hyps = model._decode_preds(preds, x_len)
refs = model._decode_targets(targets, targets_len)
for i, (ref, hyp) in enumerate(zip(refs, hyps)):
    plt.imshow(x[i, 0])
    plt.savefig(f'/tmp/ztmp{i:03d}.png')
    plt.clf()
    print(f'ztmp{i:03d}.png')
    print(f'{ref = }')
    print(f'{hyp = }')
    print('\n\n')
