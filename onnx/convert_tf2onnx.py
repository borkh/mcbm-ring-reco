import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import numpy as np
import tensorflow as tf
import subprocess
import sys

sys.path.append('..')
from train_tf import *  # nopep8
from utils.utils import *  # nopep8


@measure_time
def predict_onnx():
    result = subprocess.run(['root', '-l', '-b', '-q', 'Run_ring_finder.C'],
                            stdout=subprocess.PIPE)
    onnx_output = result.stdout.decode('utf-8')

    # Read only the third line of the output, which contains the ring parameters
    # in a single line separated by commas. Then convert the string to a numpy
    # array and reshape it to the correct shape (n, 5, 5).
    onnx_output = [line for line in onnx_output.splitlines()][2]
    onnx_output = np.fromstring(onnx_output, dtype=float, sep=',')
    onnx_output = onnx_output.reshape(len(onnx_output) // (5 * 5), 5, 5)

    return onnx_output


@measure_time
def predict_keras(model, dg):
    return model.predict(dg)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    sys.exit("No model path provided. Exiting...")

# Load the keras model
print(f'Convert keras model to onnx...')
subprocess.run(['python', '-m', 'tf2onnx.convert',
               '--saved-model', model_path, '--output', 'model.onnx'])

print(f'Loading keras model {model_path}...')
keras_model = tf.keras.models.load_model(model_path)

test_dg = dg('../data/test', batch_size=1)

print("Running keras model...")
keras_output, keras_time = predict_keras(keras_model, test_dg)
print(
    f'Keras prediction time: {keras_time:4.3f} s ({keras_time / test_dg.n:.5f} s per sample)')

print("\nRunning onnx model...")
onnx_output, onnx_time = predict_onnx()
print(
    f'ONNX prediction time: {onnx_time:5.3f} s ({onnx_time / test_dg.n:.5f} s per sample)')


abs_diff = np.abs(keras_output - onnx_output)
rel_diff = abs_diff / (np.maximum(np.abs(keras_output),
                       np.abs(onnx_output)) + 1e-12)

atol = np.amax(abs_diff)
rtol = np.amax(rel_diff)

print(f'\nMaximum absolute difference: {atol:.5f}')
print(f'Maximum relative difference: {rtol:.5f} -> {rtol * 100:.3f}%')

check_if_close = np.allclose(keras_output, onnx_output, atol=atol, rtol=rtol)
print(f'Check if arrays are close within those tolerances: {check_if_close}')
