import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import time
import plotly.express as px
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
    model_path = 'models/checkpoints/' + '1M-202212282307.model'
    model = tf.keras.models.load_model(model_path)

    sim, idealhough, hough = (x[:100] for x in hough_parameters())
    predictions, pred_time = predict(model, sim)
    print(
        f'Inference took {pred_time}s to run. {pred_time / len(sim)}s per event')

    fit_rings(sim, predictions)
