#!/usr/bin/env python3
import pickle as pkl, tensorflow as tf, time
import time

from utils.utils import *

save_loc = "data/test.pkl"
#save_loc = "data/sim+idealhough+hough+cnn_autoencoder.pkl"

def measure_time(func):
    def wrapper(*args):
        t = time.time()
        res = func(*args)
        print(f'Function took {str(time.time() - t)}s to run.')
        return res
    return wrapper


@measure_time
def predict():
    # load data
    idealhough = loadParameters("data/targets_ring_hough_ideal.csv")
    hough = loadParameters("data/targets_ring_hough.csv")
    sim = np.array(loadFeatures("data/features_denoise.csv"))

    # apply some cuts
    indices = filter_events(idealhough) # filter events with incorrectly fitted rings
    sim = sim[indices]
    hough = hough[indices]
    idealhough = idealhough[indices]

    indices = filter_events(hough) # filter events with incorrectly fitted rings
    sim = sim[indices]
    hough = hough[indices]
    idealhough = idealhough[indices]

    idealhough = idealhough.reshape(idealhough.shape[0],5,5)
    hough = hough.reshape(hough.shape[0],5,5)

    # run predictions
    model = tf.keras.models.load_model('models/autoencoder_regressor_mcbm.model')

    cnn, _ = model.predict(sim)
    print(cnn.shape)

    with open(save_loc, "wb") as f:
        pkl.dump([sim,idealhough,hough,cnn], f)


if __name__ == '__main__':
    predict()
    # load data
    idealhough = loadParameters("data/targets_ring_hough_ideal.csv")
    hough = loadParameters("data/targets_ring_hough.csv")
    sim = np.array(loadFeatures("data/features_denoise.csv"))

    # apply some cuts
    indices = filter_events(idealhough) # filter events with incorrectly fitted rings
    sim = sim[indices]
    hough = hough[indices]
    idealhough = idealhough[indices]

    indices = filter_events(hough) # filter events with incorrectly fitted rings
    sim = sim[indices]
    hough = hough[indices]
    idealhough = idealhough[indices]

    idealhough = idealhough.reshape(idealhough.shape[0],5,5)
    hough = hough.reshape(hough.shape[0],5,5)

    # run predictions
    #model = tf.keras.models.load_model('models/checkpoints/200k-final-202208071623.model')
    model = tf.keras.models.load_model('models/autoencoder_regressor_mcbm.model')
    star_time = time.time()

    #cnn = model.predict(sim)
    cnn, _ = model.predict(sim)
    print(cnn.shape)

    end_time = time.time()
    elapsed_time = end_time - star_time
    print(f'Predicting {len(sim)} events took a total of {elapsed_time}s ({elapsed_time/len(sim)}s per event)')

    with open(save_loc, "wb") as f:
        pkl.dump([sim,idealhough,hough,cnn], f)
