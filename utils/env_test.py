import subprocess
import tempfile
import os
from pathlib import Path
import sys

from utils import *  # nopep8
root_dir = Path(__file__).parent.parent

train_dir = tempfile.TemporaryDirectory()
test_dir = tempfile.TemporaryDirectory()
val_dir = tempfile.TemporaryDirectory()

train_size = 200
val_size = 50
test_size = 50


# create dummy data
for dir_, size_ in zip([train_dir, val_dir, test_dir],
                       [train_size, val_size, test_size]):
    subprocess.run(['python', str(Path(root_dir, 'data', 'create_data.py')),
                    '--target_dir', dir_.name,
                    '--n_files', str(size_), '--silent'],
                   input='y\n'.encode('utf-8'))

# find the optimal learning rate
subprocess.run(['python', str(Path(root_dir, 'train.py')),
                '--train_dir', train_dir.name,
                '--val_dir', val_dir.name,
                '--find_lr'])

# train on dummy data
subprocess.run(['python', str(Path(root_dir, 'train.py')),
                '--train_dir', train_dir.name,
                '--val_dir', val_dir.name, '--silent'])

# locate the most recent model
model_path = max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                 key=os.path.getctime)

# evaluate the model
subprocess.run(['python', str(Path(root_dir, 'evaluate.py')),
                '--model_path', model_path, '--silent'])

# convert the model to ONNX
subprocess.run(['python', str(Path(root_dir, 'onnx', 'convert_tf2onnx.py')),
                '--model_path', model_path])


for dir_ in [train_dir, val_dir, test_dir]:
    dir_.cleanup()
