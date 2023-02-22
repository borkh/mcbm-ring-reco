import cv2
import numpy as np
import pandas as pd
import torch
import time
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import sys
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)

ROOT_DIR = Path(__file__).parent
# pio.templates.default = 'presentation'
# fig_width = 700
# fig_height = 250
# margins = dict(l=90, r=5, t=5, b=65)


def measure_time(func):
    """
    Decorator to measure the time a function takes to run. Takes a function as
    input and returns the functions output and the time it took to run.
    """
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print(f'Function "{func.__name__}" took {t:.2f} seconds to run.')
        return result
    return wrapper


def plot_single_event(image, Y1=None, Y2=None, Y3=None,
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
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.repeat(image, 3, axis=2)

    image = cv2.resize(image, (image.shape[1]*scaling,
                               image.shape[0]*scaling),
                       interpolation=cv2.INTER_AREA)
    colors = [(0., 1., 1.), (1., 0., 0.), (1., 1., 0.), (0., 1., 1.)]
    if type(Y1) == torch.Tensor:
        Y_values = [y.cpu().numpy() for y in [Y1, Y2, Y3, Y4] if y is not None]
    else:
        Y_values = [Y1, Y2, Y3, Y4]

    for Y, color in zip(Y_values, colors):  # type: ignore
        if Y is not None:
            for ring in (Y*scaling).astype(int):
                try:
                    image = cv2.ellipse(image, (ring[0], ring[1]),
                                        (ring[2], ring[3]),
                                        0, 0, 360, color, 2)
                except cv2.error as e:
                    print(e, f': {ring}')

    return image


def load_sim_data() -> None:
    """
    This function loads simulation data from a csv file, filters the data based
    on the filters below, sorts the order of the rings in the data, and creates
    a dataset of images and labels in the same format as the toymodel datasets
    `data/train`, `data/val`, and `data/test`.

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
    """
    # img_dir = ROOT_DIR / 'data' / 'sim_data' / 'X'
    df = pd.read_csv(ROOT_DIR / 'data' / 'sim_data' /
                     'ring_hough_idealhough.csv')
    # remove nan and inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # remove rows where x, y coordinates are out of bounds or radius is too large
    for i in range(10):
        df = df.iloc[np.where((df.iloc[:, i*5] >= 0.) &
                              (df.iloc[:, i*5] <= 72.) &
                              (df.iloc[:, i*5 + 1] >= 0.) &
                              (df.iloc[:, i*5 + 1] <= 72.) &
                              (df.iloc[:, i*5 + 2] <= 10.))[0]]

    # remove rows where all values are 0
    df = df.iloc[np.where((df.iloc[:, 0:25] != 0.).any(axis=1))[0]]
    df = df.iloc[np.where((df.iloc[:, 25:50] != 0.).any(axis=1))[0]]

    # get row indices of df
    indices = df.index.values

    sim_df = pd.read_csv(ROOT_DIR / 'data' / 'sim_data' / 'mrich_events.csv')
    sim_df = sim_df.iloc[indices]

    X = sim_df.to_numpy().reshape(-1, 72, 32, 1)
    # get ideal hough parameters only
    y = df.iloc[:, :25].to_numpy().reshape(-1, 5, 5)

    for i in range(5):
        # add indices to df with dtype int
        df[f'indices{i}'] = df[f'x{i}'].astype(
            int) + df[f'y{i}'].astype(int) * 32

    # check where indices0 is bigger than indices1 and x0 is not 0
    df['check'] = np.where((df['indices0'] > df['indices1']) &
                           (df['x1'] != 0), 1, 0)
    # swap first five columns with second five columns if check is 1
    tmp = df.loc[df['check'] == 1, df.columns[:5]].values  # type: ignore
    df.loc[df['check'] == 1, df.columns[:5]
           ] = df.loc[df['check'] == 1, df.columns[5:10]].values  # type: ignore
    df.loc[df['check'] == 1, df.columns[5:10]] = tmp

    # add indices to df in order to check if they are in the correct order
    for i in range(5):
        df[f'indices{i}'] = df[f'x{i}'].astype(
            int) + df[f'y{i}'].astype(int) * 32
    df['check'] = np.where((df['indices0'] > df['indices1']) &
                           (df['x1'] != 0), 1, 0)

    # drop check and indices columns
    df = df.drop(columns=['check', 'indices0', 'indices1', 'indices2',
                          'indices3', 'indices4'])

    # create dataset in data/sim_data
    target_X = ROOT_DIR / 'data' / 'sim_data' / 'X'
    target_y = ROOT_DIR / 'data' / 'sim_data' / 'y'
    print(f'Creating dataset in {target_X} and {target_y}...')

    for i, (x, y) in enumerate(zip(X, y)):
        im_path = target_X / f'{i}.png'
        label_path = target_y / f'{i}.npy'
        cv2.imwrite(str(im_path), 255*x)
        np.save(str(label_path), y)


def ring_params_hist(y, silent=False):
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

    fig.add_trace(go.Histogram(x=df['n_rings'],
                  histnorm='probability'), row=5, col=1)

    fig.update_layout(width=1000, modebar_add=["toggleSpikelines"],
                      margin=dict(l=0.2, r=0.2, t=50.0, b=0.2))

    if not silent:
        fig.show()

    return fig


def display_images(imgs: np.ndarray, col_width: int = 5, title="", silent=False) -> np.ndarray:
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
        if not silent:
            fig.show()
    except ValueError as e:
        print(f'\nError: {e}')
        print(f'The number of images must be at least {col_width}.')
    finally:
        return imgs


def fit_rings(images, params, title="", silent=False) -> np.ndarray:
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
    return display_images(ring_fits, title=title, silent=silent)
