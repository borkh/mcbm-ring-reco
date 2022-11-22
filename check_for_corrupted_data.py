import os

import cv2
import numpy as np
import plotly.express as px
from tqdm import tqdm

from train_tf import dg

files = os.listdir('data/train/X')
files.remove('.gitignore')


bs = 500
spe = len(files) // bs
gen = dg('data/train', batch_size=bs)

for i in tqdm(range(len(spe))):
    X, y = gen.__getitem__(i)
