import shutil
import os

import cv2
import numpy as np
import plotly.express as px
from tqdm import tqdm

from train_tf import dg


"""
files = os.listdir('data/train/X')
files.remove('.gitignore')

bs = 1
spe = len(files) // bs
gen = dg('data/train', batch_size=bs)

for i in tqdm(range(spe)):
    X, y = gen.__getitem__(i)
"""


src = 'data/train/'
dst = 'data/train300k/'

os.makedirs(dst + 'X', exist_ok=True)
os.makedirs(dst + 'y', exist_ok=True)

for i in tqdm(range(300000)):
    shutil.copy(f'{src}X/{i}.png', f'{dst}X/{i}.png')
    shutil.copy(f'{src}y/{i}.npz', f'{dst}y/{i}.npz')
