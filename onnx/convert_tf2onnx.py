import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import numpy as np
import tensorflow as tf
import subprocess
import sys
import argparse
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.utils import *  # nopep8
from data.create_data import DataGen  # nopep8
from models.model import custom_loss  # nopep8


@measure_time
def predict_onnx():
    cwd = os.getcwd()
    onnx_dir = Path(__file__).parent
    os.chdir(onnx_dir)
    result = subprocess.run(['root', '-l', '-b', '-q', 'Run_ring_finder.C'],
                            stdout=subprocess.PIPE)
    onnx_output = result.stdout.decode('utf-8')

    # Read only the third line of the output, which contains the ring parameters
    # in a single line separated by commas. Then convert the string to a numpy
    # array and reshape it to the correct shape (n, 5, 5).
    onnx_output = [line for line in onnx_output.splitlines()][2]
    onnx_output = np.fromstring(onnx_output, dtype=float, sep=',')
    onnx_output = onnx_output.reshape(len(onnx_output) // (5 * 5), 5, 5)
    os.chdir(cwd)

    return onnx_output


def run(model_path, test_dir=Path(root_dir, 'data', 'test')):
    # Load the keras model
    onnx_model_path = Path(root_dir, 'models', 'model.onnx')
    print(
        f'Converting keras model {model_path} to onnx model {onnx_model_path}...')
    subprocess.run(['python', '-m', 'tf2onnx.convert',
                    '--saved-model', model_path, '--output', onnx_model_path])

    print(f'Loading keras model {model_path}...')
    keras_model = tf.keras.models.load_model(model_path,
                                             custom_objects={'custom_loss': custom_loss})

    test_gen = DataGen(test_dir, batch_size=1000)

    try:
        subprocess.run(['root', '-q'], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except FileNotFoundError:
        exit_msg = ('Unable to run further tests because ROOT installation was not found. '
                    'Please install or source ROOT before running this script. Exiting...')
        sys.exit(exit_msg)
    else:
        print("Running keras model...")
        keras_output, keras_time = predict(keras_model, test_gen)
        print(
            f'Keras prediction time: {keras_time:4.3f} s ({keras_time / test_gen.n:.5f} s per sample)')

        print("\nRunning onnx model...")
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

        check_if_close = np.allclose(keras_output, onnx_output,
                                     atol=atol, rtol=rtol)
        print(
            f'Check if arrays are close within those tolerances: {check_if_close}')

    return keras_output, onnx_output


if __name__ == '__main__':
    """
    Run this script to convert a keras model to onnx format. After
    conversion, the onnx model will be run and the output will be compared
    with the output of the keras model. The prediction time of both models
    will also be printed.

    Arguments
    ---------
        model_path : str
            Path to the model that will be converted to onnx format. (e.g.
            models/checkpoints/4M-202212072326.model). If not specified, the
            most recent model in the models/checkpoints directory will be used.
        test_dir : str
            Path to the test data directory. (e.g. data/test). If not
            specified, the `data/test` directory will be used.
    """
    try:
        __IPYTHON__  # type: ignore
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str,
                            default=max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                                        key=os.path.getctime),
                            help='''Path to the model that will be converted to
                            onnx format.''')
        parser.add_argument('--test_dir', type=str,
                            default=Path(root_dir, 'data', 'test'),
                            help='''Path to the test data directory. (e.g.
                            data/test)''')
        args = parser.parse_args()
        model_path = Path(args.model_path).resolve()
        test_dir = Path(args.test_dir).resolve()
    else:
        # load the latest model
        model_path = max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                         key=os.path.getctime)
        test_dir = Path(root_dir, 'data', 'test')

    keras_output, onnx_output = run(model_path, test_dir)
