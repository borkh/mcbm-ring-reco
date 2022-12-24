import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import plotly.express as px


def measure_time(func):
    def wrapper(*args):
        t = time.time()
        res = func(*args)
        t = time.time() - t
        return res, t
    return wrapper


def plot_single_event(image: np.ndarray, Y1=None, Y2=None, Y3=None,
                      Y4=None, scaling: int = 10) -> np.ndarray:
    """
    Plot ellipses on an image.

    Parameters
    ----------
        image : np.ndarray
            The image to plot on.
        Y1, Y2, Y3, Y4 : Optional[np.ndarray]
            An array of ellipses to plot on the image, with each ellipse
            represented as a 5-element tuple (x, y, major axis, minor axis,
            angle). The ellipses will be plotted greeen (Y1), red (Y2), Yellow
            (Y3), and/or cyan (Y4).
        scaling : int, optional
            The scaling factor to use when resizing the image.

    Returns
    -------
        np.ndarray
            The modified image with ellipses plotted on it.
    """

    image = cv2.resize(image, (image.shape[1]*scaling,
                               image.shape[0]*scaling),
                       interpolation=cv2.INTER_AREA)
    colors = [(0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]
    Y_values = [Y1, Y2, Y3, Y4]

    for Y, color in zip(Y_values, colors):
        if Y is not None:
            for ring in (Y*scaling).astype(int):
                try:
                    image = cv2.ellipse(image, (ring[0], ring[1]),
                                        (ring[2], ring[3]),
                                        0, 0, 360, color, 2)
                except cv2.error as e:
                    print(e)

    return image


def display_images(imgs: np.ndarray, col_width: int = 5) -> None:
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)
    imgs = imgs[:len(imgs) // col_width * col_width]
    imgs = imgs.reshape(imgs.shape[0] // col_width, col_width, *imgs.shape[1:])
    fig = px.imshow(imgs, animation_frame=0, facet_col=1, binary_string=True,
                    height=700)
    fig.show()


def fit_rings(images, params) -> None:
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    ring_fits = np.array([plot_single_event(x, y)
                          for x, y in zip(images, params)])
    display_images(ring_fits)


# functions for reading .csv file from simulation data
def loadFeatures(datafile, pixel_x=32, pixel_y=72):
    with open(datafile, 'r') as temp_f:
        col_count = [len(l.split(",")) for l in temp_f.readlines()]
    column_names = [i for i in range(0, max(col_count))]
    hits = pd.read_csv(datafile, header=None, index_col=0, comment='#',
                       delimiter=",", nrows=20000, names=column_names).values.astype('int32')  # type: ignore
    hits[hits < 0] = 0
    hits_temp = np.zeros([len(hits[:, 0]), pixel_x*pixel_y])
    for i in range(len(hits[:, 0])):
        for j in range(len(hits[0, :])):
            if hits[i, j] == 0:
                break
            hits_temp[i, hits[i, j]-1] += 1
    hits_temp = tf.reshape(hits_temp, [len(hits[:, 0]), pixel_y, pixel_x])
    hits_temp = tf.clip_by_value(
        hits_temp, clip_value_min=0., clip_value_max=1.)
    hits = tf.cast(hits_temp[..., tf.newaxis],  # type: ignore
                   dtype=tf.float32)
    print('load data from  ' + datafile + '  -> ' +
          str(len(hits[:])) + '  events loaded')  # type: ignore
    return hits


def loadParameters(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
        n = len(lines)
    params = np.zeros((n, 25))
    for i, line in enumerate(lines):
        line = line.strip().split(",")
        line.remove("")
        line = np.array([float(x) for x in line])
        for j, par in enumerate(line):
            try:
                params[i, j] = np.round(par, 2)
            except IndexError as e:
                print(e)
    return params


def filter_events(y: np.ndarray) -> np.ndarray:
    """Filter events based on certain criteria.

    This function filters events in a dataset based on the following criteria:
    - Removing events with NaN (Not a Number) in any of the parameters
    - Removing events with all zero values in the parameters
    - Removing events where the x-coordinate of the first ring is less than 0 or
      greater than 72
    - Removing events where the y-coordinate of the first ring is less than 0 or
      greater than 72
    - Removing events where the radius of the first ring is greater than 20
    - Repeating the above steps for the second through fifth rings
      (x-coordinate, y-coordinate, radius).

    Parameters
    ----------
    y : numpy array
        A 2D array of shape (n_events, n_parameters) containing the parameters
        for each event. The parameters represent the x-coordinate, y-coordinate,
        semi-major axis, semi-minor axis, and rotation angle of ellipses.

    Returns
    -------
    filtered_events : numpy array
        A 2D array with the same shape as `y`, but potentially fewer rows,
        containing the events that meet the filtering criteria.
    """
    cond1 = np.all(~np.isnan(y), axis=1)
    cond2 = np.invert(np.all(y == 0., axis=1))

    filtered_events = np.where(cond1 & cond2)[0]

    for i in range(5):
        filtered_events = np.intersect1d(filtered_events,
                                         np.where((y[:, i*5] >= 0.) &
                                                  (y[:, i*5] <= 72.) &
                                                  (y[:, i*5 + 1] >= 0.) &
                                                  (y[:, i*5 + 1] <= 72.) &
                                                  (y[:, i*5 + 2] <= 20.))[0])
    print(f'Filtered {len(y) - len(filtered_events)} events out of {len(y)} events.',
          f'\nRemaining events: {len(filtered_events)}')

    return filtered_events
