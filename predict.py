import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import argparse
from utils.utils import *


@measure_time
def predict(model, simulation_data):
    return model.predict(simulation_data)


def hough_parameters():
    # load simulation data
    idealhough = loadParameters("data/sim_data/targets_ring_hough_ideal.csv")
    hough = loadParameters("data/sim_data/targets_ring_hough.csv")
    sim = np.array(loadFeatures("data/sim_data/features_denoise.csv"))

    # apply some cuts on both idealhough and hough
    indices = filter_events(idealhough)
    indices = filter_events(hough[indices])

    sim, idealhough, hough = sim[indices], idealhough[indices], hough[indices]

    idealhough = idealhough.reshape(idealhough.shape[0], 5, 5)

    hough = hough.reshape(hough.shape[0], 5, 5)

    return sim, idealhough, hough


if __name__ == '__main__':
    try:
        __IPYTHON__
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, required=True,
                            help='''Path to the model that will be converted to
                            ONNX format.''')
        parser.add_argument('--nof_plots', type=int, default=200,
                            help='''Number of plots to be generated.''')
        args = parser.parse_args()
        model_path = os.path.abspath(args.model_path)
        nof_plots = args.nof_plots
    else:
        model_path = 'models/checkpoints/' + '10M-202212311744.model'
        nof_plots = 200

    model = tf.keras.models.load_model(model_path)

    sim, idealhough, hough = (x[:nof_plots] for x in hough_parameters())
    predictions, pred_time = predict(model, sim)
    print(
        f'Inference took {pred_time}s to run. {pred_time / len(sim)}s per event')

    fit_rings(sim, predictions)
