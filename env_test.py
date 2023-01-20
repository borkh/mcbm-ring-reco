import subprocess
from pathlib import Path
import tempfile
from utils.utils import *
from data.create_data import DataGen
import os
import glob

train_dir = tempfile.TemporaryDirectory()
test_dir = tempfile.TemporaryDirectory()
val_dir = tempfile.TemporaryDirectory()

train_size = 200
val_size = 50
test_size = 50

try:
    for dir_, size_ in zip([train_dir, val_dir, test_dir],
                        [train_size, val_size, test_size]):
        subprocess.run(['python', 'data/create_data.py',
                        '--target_dir', dir_.name,
                        '--n_files', str(size_), '--silent'],
                    input='y\n'.encode('utf-8'))

    # subprocess.run(['python', 'train.py',
    #         '--train_dir', train_dir.name,
    #         '--val_dir', val_dir.name,
    #         '--find_lr'])

    subprocess.run(['python', 'train.py',
                    '--train_dir', train_dir.name,
                    '--val_dir', val_dir.name])

    # locate the most recent model
    model_path = max(glob.glob('models/checkpoints/*.model'), key=os.path.getctime)

    subprocess.run(['python', 'evaluate.py',
                    '--model_path', model_path])

    subprocess.run(['python', 'onnx/convert_tf2onnx.py',
                    '--model_path', model_path])
except Exception as e:
    print(e)

for dir_ in [train_dir, val_dir, test_dir]:
    dir_.cleanup()
