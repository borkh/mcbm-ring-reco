import subprocess
import os
import sys
from pathlib import Path


root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from utils import *  # nopep8

script_dir = root_dir / "onnx"
os.chdir(script_dir)

"""
This file runs the mcbm_reco_rich_cnn.C script and captures the output.
The output is then parsed and used to visualize the model performance by
creating ring fits.
"""


result = subprocess.run(
    ["root", "-b", "-q", "mcbm_reco_rich_cnn.C"], stdout=subprocess.PIPE)
output = result.stdout.decode('utf-8')

# read only data after following line
start_point = '-I- mcbm_reco: Starting run'
end_point = '\n\n'
output = output[output.find(start_point) + len(start_point):]
output = output[:output.find(end_point)]

output = [line for line in output.splitlines() if line.strip()]


# write odds and evens to separate variables: imgs and labels
imgs = [line for i, line in enumerate(output) if i % 2 == 0]
labels = [line for i, line in enumerate(output) if i % 2 == 1]

imgs_arr = np.array([np.fromstring(img, sep=',', dtype=np.float32)
                    for img in imgs]).reshape(-1, 72, 32, 1)
labels_arr = np.array([np.fromstring(label, sep=',')
                      for label in labels]).reshape(-1, 5, 5)

fit_rings(imgs_arr, labels_arr)
