import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import numpy as np
import tensorflow as tf
import subprocess
import sys
import argparse

sys.path.append('.')
sys.path.append('..')
from utils.utils import *  # nopep8
from data.create_data import DataGen  # nopep8


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


def run(model_path):
    # Load the keras model
    subprocess.run(['python', '-m', 'tf2onnx.convert',
                    '--saved-model', model_path, '--output', '../models/model.onnx'])

    print(f'Loading keras model {model_path}...')
    keras_model = tf.keras.models.load_model(model_path)

    test_gen = DataGen('../data/test', batch_size=2000)

    print("Running keras model...")
    keras_output, keras_time = predict(keras_model, test_gen)
    print(
        f'Keras prediction time: {keras_time:4.3f} s ({keras_time / test_gen.n:.5f} s per sample)')

    print("\nRunning onnx model...")
    try:
        onnx_output, onnx_time = predict_onnx()
        print(
            f'ONNX prediction time: {onnx_time:5.3f} s ({onnx_time / test_gen.n:.5f} s per sample)')

        abs_diff = np.abs(keras_output - onnx_output)
        rel_diff = abs_diff / (np.maximum(np.abs(keras_output),
                                          np.abs(onnx_output)) + 1e-12)

        atol = np.amax(abs_diff)
        rtol = np.amax(rel_diff)

        print(f'\nMaximum absolute difference: {atol:.5f}')
        print(f'Maximum relative difference: {rtol:.5f} -> {rtol * 100:.3f}%')

        check_if_close = np.allclose(
            keras_output, onnx_output, atol=atol, rtol=rtol)
        print(
            f'Check if arrays are close within those tolerances: {check_if_close}')
    except Exception as e:
        print(e)
        onnx_output = None

    return keras_output, onnx_output


if __name__ == '__main__':
    try:
        __IPYTHON__  # type: ignore
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, required=True,
                            help='''Path to the model that will be converted to
                            onnx format. (e.g.
                            models/checkpoints/4M-202212072326.model)''')
        args = parser.parse_args()
        model_path = os.path.abspath(args.model_path)
    else:
        model_path = '../models/checkpoints/10M-202212311744.model'

    script_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(script_path))

    keras_output, onnx_output = run(model_path)
