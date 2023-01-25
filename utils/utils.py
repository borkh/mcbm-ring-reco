import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio
import sys
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)

root_dir = Path(__file__).parent.parent
pio.templates.default = 'presentation'


def measure_time(func):
    """
    Decorator to measure the time a function takes to run. Takes a function as
    input and returns the functions output and the time it took to run.
    """
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        return result, t
    return wrapper


@measure_time
def predict(model, X):
    return model.predict(X)


@measure_time
def evaluate(model, data_generator):
    return model.evaluate(data_generator)


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

    for Y, color in zip(Y_values, colors):  # type: ignore
        if Y is not None:
            for ring in (Y*scaling).astype(int):
                try:
                    image = cv2.ellipse(image, (ring[0], ring[1]),
                                        (ring[2], ring[3]),
                                        0, 0, 360, color, 2)
                except cv2.error as e:
                    print(e)

    return image


def display_images(imgs: np.ndarray, plot_dir=None, col_width: int = 5, title="", silent=False) -> None:
    """
    Display a set of images in a grid.

    Parameters
    ----------
        imgs : np.ndarray
            The images to display.
        col_width : int, optional
            The number of images to display per row.
        title : str, optional
            The title to display above the grid of images.
    """
    try:
        if imgs.shape[-1] == 1:
            imgs = np.repeat(imgs, 3, axis=-1)
        imgs = imgs[:len(imgs) // col_width * col_width]
        imgs = imgs.reshape(
            imgs.shape[0] // col_width, col_width, *imgs.shape[1:])
        fig = px.imshow(imgs, animation_frame=0, facet_col=1, title=title)
        if plot_dir is not None:
            plot_path = plot_dir / f'{title}.html'
            pio.write_html(fig, plot_path, auto_open=not silent)
        elif not silent:
            fig.show()
    except ValueError as e:
        print(f'\nError: {e}')
        print(f'The number of images must be at least {col_width}.')


def fit_rings(images, params, plot_dir=None, title="", silent=False) -> None:
    """
    Uses the `plot_single_event` function to fit rings to a set of images
    with the given parameters. The resulting images are then displayed in a
    grid using the `display_images` function.

    Parameters
    ----------
        images : np.ndarray
            The images to fit rings to.
        params : np.ndarray
            The ring parameters to fit to the images. The shape of this array
            should be (N, 5, 5), where N is the number of images, which each
            can have up to 5 rings, each represented as a 5-element tuple
            (x, y, major axis, minor axis, angle).
        title : str, optional
            The title to display above the grid of images.
        silent : bool, optional
            If True, the images will not be displayed.
    """
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    ring_fits = np.array([plot_single_event(x, y)
                          for x, y in zip(images, params)])
    display_images(ring_fits, plot_dir, title=title, silent=silent)


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
    print('Loading data from  ' + str(datafile) + '  -> ' +
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


def load_sim_data():
    """
    This function loads the simulated data from the data/sim_data directory.
    Some cuts are applied to the data to remove events that contain bad ring
    parameters.

    Returns
    -------
    sim : numpy array
        The simulated images.
    idealhough : numpy array
        The ideal Hough transform parameters for each image.
    hough : numpy array
        The Hough transform parameters for each image.
    """
    idealhough_path = str(root_dir / 'data' / 'sim_data' /
                          'targets_ring_hough_ideal.csv')
    idealhough = loadParameters(idealhough_path)
    hough_path = str(root_dir / 'data' / 'sim_data' / 'targets_ring_hough.csv')
    hough = loadParameters(hough_path)
    sim_path = str(root_dir / 'data' / 'sim_data' / 'features_denoise.csv')
    sim = np.array(loadFeatures(sim_path))

    # apply some cuts on both idealhough and hough
    indices = filter_events(idealhough)
    indices = filter_events(hough[indices])

    sim, idealhough, hough = sim[indices], idealhough[indices], hough[indices]

    idealhough = idealhough.reshape(idealhough.shape[0], 5, 5)
    hough = hough.reshape(hough.shape[0], 5, 5)

    return sim, idealhough, hough


def ring_params_hist(y, plot_dir=None, title='Ring Parameters Histograms', silent=False):
    """
    This function creates a pd.DataFrame from the ring parameters and plots
    histograms of each parameter. These might be useful for visualizing the
    distribution of the parameters.

    Parameters
    ----------
    y : numpy array
        A 2D array of shape (n_events, n_parameters) containing the parameters
        for each event. The parameters represent the x-coordinate, y-coordinate,
        semi-major axis, semi-minor axis, and rotation angle of ellipses.
    title : str
        The title of the plot.
    """
    cols = ['x_0', 'y_0', 'a_0', 'b_0', 'theta_0',
            'x_1', 'y_1', 'a_1', 'b_1', 'theta_1',
            'x_2', 'y_2', 'a_2', 'b_2', 'theta_2',
            'x_3', 'y_3', 'a_3', 'b_3', 'theta_3',
            'x_4', 'y_4', 'a_4', 'b_4', 'theta_4']

    df = pd.DataFrame(y.reshape(-1, 25), columns=cols)
    df['n_rings'] = df[cols].apply(lambda x: np.sum(x != 0)//4, axis=1)

    # add a column for the number of rings
    columns = cols[:20]
    columns.append('n_rings')
    # Create a subplot layout with 5 rows and 5 columns
    fig = sp.make_subplots(rows=5, cols=5, subplot_titles=columns)

    # iterate through the range
    for i in range(4):
        # Create a histogram for x_i column and add it to the i-th row 1st column
        fig.add_trace(go.Histogram(
            x=df[f'x_{i}'], xbins=dict(start=0.1)), row=i+1, col=1)
        # Create a histogram for y_i column and add it to the i-th row 2nd column
        fig.add_trace(go.Histogram(
            x=df[f'y_{i}'], xbins=dict(start=0.1)), row=i+1, col=2)
        # Create a histogram for a_i column and add it to the i-th row 3rd column
        fig.add_trace(go.Histogram(
            x=df[f'a_{i}'], xbins=dict(start=0.1)), row=i+1, col=3)
        # Create a histogram for b_i column and add it to the i-th row 4th column
        fig.add_trace(go.Histogram(
            x=df[f'b_{i}'], xbins=dict(start=0.1)), row=i+1, col=4)
        # Create a histogram for theta_i column and add it to the i-th row 5th column
        fig.add_trace(go.Histogram(
            x=df[f'theta_{i}'], xbins=dict(start=0.1)), row=i+1, col=5)

    fig.add_trace(go.Histogram(x=df['n_rings']), row=5, col=1)

    fig.update_layout(title=title, width=1000,
                      modebar_add=["toggleSpikelines"])

    if plot_dir is not None:
        plot_path = plot_dir / f'{title}.html'
        print(f'Saving plot to {plot_path}...')
        pio.write_html(fig, plot_path, auto_open=not silent)
    elif not silent:
        fig.show()


def plot_lr_range(lr_finder, plot_dir, n_skip_beginning=20, n_skip_end=3, silent=False):
    """
    This function plots the loss vs learning rate for the learning rate range
    test.

    Parameters
    ----------
    lr_finder : lr_finder.LRFinder
        The learning rate finder object.
    n_skip_beginning : int
        The number of batches to skip at the beginning.
    n_skip_end : int
        The number of batches to skip at the end.
    """
    fig = px.line(x=lr_finder.lrs[n_skip_beginning:-n_skip_end],
                  y=lr_finder.losses[n_skip_beginning:-n_skip_end],
                  labels={'x': 'Learning Rate (log scale)', 'y': 'Loss'},
                  log_x=True, title='Loss vs Learning Rate')
    fig.update_layout(modebar_add=["toggleSpikelines"])
    fig.update_xaxes(exponentformat='power')
    plot_path = plot_dir / 'lr_range.html'
    print(f'Saving plot to {plot_path}...')
    pio.write_html(fig, plot_path, auto_open=not silent)


def hits_on_ring(y_true, y_pred):
    """
    This function calculates the number of hits for each ring in each event.  It
    was used to calculate the number of hits that are on the ring for each ring
    in each event, in order to add a penalty to a custom loss function if the
    number of hits on the ring is less than a certain threshold. It worked as
    intended, but using this function in the loss function caused training to
    be extremely slow.

    Might be useful for future work. If you want to use it for a custom loss
    function, you might need to add 'run_eagerly=True' to 'model.compile()' in
    the 'train.py' file. Alternatively, this function can be edited to
    use tensors and tensorflow operations instead of numpy arrays and numpy
    operations, which should also make it faster.

    Parameters
    ----------
    y_true : numpy array
        The true values of the hits on the ring.
    y_pred : numpy array
        The predicted values of the hits on the ring.

    Returns
    -------
    np.sum(penalty) : float
        The sum of the penalty for each event.
    """
    pars = y_pred[..., :5]
    hits = y_true[..., 5:]
    hits = hits.reshape((y_pred.shape[0], y_pred.shape[1], -1, 2))

    # check the number of hits on the ring for each ring in each event
    # and store it in n_hits
    n_hits = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    for i, j, k in np.ndindex((y_pred.shape[0], y_pred.shape[1], (y_pred.shape[2]-5) // 2)):
        hits_ = hits[i, j, k]
        pars_ = pars[i, j]
        if np.all(hits_.numpy() != 0.):
            # equal_zero = tf.reduce_all(tf.equal(hits_, tf.constant(0.)))
            # if not equal_zero:
            x = hits_[1] + 0.5 - pars_[0]
            y = hits_[0] + 0.5 - pars_[1]
            if np.isclose(np.sqrt(x**2 + y**2), pars_[2], atol=1.5):
                n_hits[i, j] += 1

    # if there are less than 10 hits on a ring with ring parameters
    # that are not all zero, add a penalty of 1 to the loss
    penalty = np.zeros((y_pred.shape[0], y_pred.shape[1]))
    for i, j in np.ndindex((y_pred.shape[0], y_pred.shape[1])):
        if n_hits[i, j] < 7 and not np.all(y_true[i, j, :5] == 0):
            penalty[i, j] = 1

    return np.sum(penalty)
