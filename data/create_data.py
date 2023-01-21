import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8

import argparse

import numpy as np
import tensorflow as tf
from numpy.random import choice
from sklearn.datasets import make_circles
from tqdm import tqdm

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.utils import *  # nopep8


class DataGen(tf.keras.utils.Sequence):
    """
    A custom data generator class for loading images and labels from a
    directory.

    This class extends the Sequence class from tf.keras.utils, allowing it to be
    used for training and evaluation of a model using the fit method
    in tf.keras.Model.

    Attributes:
        target_dir (str): The directory containing the images and labels.
        batch_size (int): The batch size.
        n (int): The total number of images in the directory.

    Methods:
        init(self, target_dir, batch_size=32): Initializes the data generator
            with the target directory and batch size.
        getitem(self, index): Retrieves a batch of images and labels at the
            given index.
        len(self): Returns the number of batches in the data generator.
    """

    def __init__(self, target_dir, batch_size=32):
        self.target_dir = tf.compat.as_str_any(target_dir)
        self.batch_size = batch_size
        self.n = len(list(Path(self.target_dir, 'X').glob('*.png')))
        self.current_index = 0

    def __iter__(self):
        return self  # return the iterator object

    def __next__(self):
        if self.current_index >= self.n:
            raise StopIteration
        else:
            result = self.__getitem__(self.current_index)
            self.current_index += 1
            return result

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        X = []
        y = []
        for i in range(start, end):
            x_path = Path(self.target_dir, 'X', f'{i}.png')
            y_path = Path(self.target_dir, 'y', f'{i}.npy')
            X.append(cv2.imread(str(x_path), cv2.IMREAD_GRAYSCALE)  # type: ignore
                     [:, :, np.newaxis] / 255.)
            y.append(np.load(str(y_path)))
        return np.array(X), np.array(y)

    def __len__(self):
        return self.n//self.batch_size


class Display(np.ndarray):
    params: np.ndarray
    positions: np.ndarray
    nof_hits_to_delete: int
    ring_hits: np.ndarray

    """
    A subclass of `numpy.ndarray` representing a display.
    
    This class extends `numpy.ndarray` with additional attributes and methods to
    simulate a mCBM event display with ellipses and noise hits. Hits denote
    pixels on the display that are firing and are marked with the value 1
    (white).
    
    Attributes:
        params (np.ndarray): An array of shape (5, 5) initialized to all zeros.
            This will be used to store the parameters of ellipses that are
            generated on the display. The first dimension represents the number
            of ellipses, and the second dimension represents the parameters of
            each ellipse. The parameters are as follows:
                [x, y, semi-major axis, semi-minor axis, angle]
            For now only perfect rings are supported, so the semi-major and
            semi-minor axes are the same and the angle is 0:
                [x, y, radius, radius, 0]
        positions (np.ndarray): An array of all the positions (x, y) on the
            display object, used to check if ellipse points are inside the display
            or not.
        nof_hits_to_delete (int): An integer initialized to 20. This will be used
            add hits to an ellipse and then delete them later again in order to
            create ellipses with irregular distances between hits.
        info (Any): An attribute that can be set to any value when creating a
            Display object. It is initialized to None by default.
    
    Methods:
        _add_random_noise(nof_noise_hits=0): Adds a specified number of randomly
            placed hits (1's) to the display object.
        _get_indices(nof_rings): Generates a specified number of random indices
            on the display, with the condition that the positions corresponding to
            the indices are at least 5 units apart.
        _init_ellipse(xshift: int, yshift: int, r: int, rn: float, hits: int):
            Generates a circle with a given radius and a specified amount of noise,
            and shifts it to a specified position.
        _add_ellipse(*args): Adds an ellipse with a specified number of hits, rotation, and semi-axes to the display object.
    """

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        obj = super().__new__(Display, shape, dtype, buffer=np.zeros(shape),
                              offset=offset, strides=strides, order=order)
        obj.params = np.zeros((5, 5))
        obj.nof_rings = 0
        obj.ring_hits = np.array([[-1, -1]])
        obj.positions = np.array([(x, y) for x in range(obj.shape[0])
                                  for y in range(obj.shape[1])])
        obj.nof_hits_to_delete = 20
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    def _add_random_noise(self, nof_noise_hits=0):
        for _ in range(nof_noise_hits):
            x, y = np.random.randint(
                0, self.shape[0]), np.random.randint(0, self.shape[1])
            self[x, y] = 1

    def _get_indices(self, nof_rings: int) -> list:
        """
        Generate a specified number of random indices on the display, with the condition that the
        positions (x, y) corresponding to the indices are at least 10 units apart in euclidean distance.
        This is done to ensure that the rings are not too close to each other.
        The indices are then used to create the rings.

        Parameters:
            nof_rings (int): The number of rings to generate indices for.

        Returns:
            list: A list of indices representing the ring centers on the display, sorted in ascending order.
        """

        self[:, :] = 0  # set to zeroes in case of multiple calls
        min_dist = 10
        indices = []
        break_con = 0
        while len(indices) < nof_rings:
            x, y = np.random.randint(
                0, self.shape[0]), np.random.randint(0, self.shape[1])
            if break_con > 100:
                indices = []
                break_con = 0
            if len(indices) == 0:
                indices.append(x*self.shape[1]+y)
            else:
                if all([np.linalg.norm(np.array([x, y]) - np.array(divmod(i, self.shape[1]))) > min_dist for i in indices]):  # type: ignore
                    indices.append(x*self.shape[1]+y)
                else:
                    break_con += 1

        return sorted(indices)

    def _init_ellipse(self, xshift: int, yshift: int, radius: int,
                      radius_noise: float, hits: int) -> np.ndarray:
        """
        Generate an ellipse template. The points of this ellipse are then used in
        `_add_ellipse` to generate the actual ellipse.

        Parameters:
            xshift (int): The number of units to shift the ellipse along the x-axis.
            yshift (int): The number of units to shift the ellipse along the y-axis.
            radius (int): The radius of the ellipse.
            radius_noise (float): The amount of noise to add to the ellipse. A
                value of 0.0 means no noise, and a value of 1.0 means maximum noise.
            hits (int): The number of hits (1's) to add to the ellipse.

        Returns:
            numpy.ndarray: An ellipse template with the specified parameters.
        """
        ellipse_template = np.array(make_circles(noise=radius_noise, factor=.1,
                                                 n_samples=(hits, 0))[0])  # type: ignore

        # scale the ellipse to the desired radius and shift it to the desired position
        ellipse_template[:, 1] = ellipse_template[:, 1] * radius + xshift
        ellipse_template[:, 0] = ellipse_template[:, 0] * radius + yshift

        ellipse_template = np.round(ellipse_template, 0).astype('int8')

        return ellipse_template

    def _add_ellipse(self, *args) -> np.ndarray:
        """
        Add an ellipse with a specified number of hits, radius, and noise to the
        display object, and return the ellipse points.

        The ellipse is generated using the `_init_ellipse` method and is scaled
        based on the radius. The number of hits is determined by sampling from a
        uniform distribution between the values specified in `hits_per_rings`.
        The ellipse is then modified to remove hits that are outside of the
        display or are duplicates, and to ensure that it has at least 12 hits.

        Parameters:
            *args: A tuple of arguments passed to the method. The first four
                arguments should be the horizontal and vertical shifts of the
                center of the ellipse, the radius, and the amount of noise to
                add to the ellipse. The fifth argument should be the number of
                hits to add to the ellipse.

        Returns:
            np.ndarray: An array of shape (n, 2) representing the positions of
                the ellipse points, where n is the number of points.
        """
        hits_per_rings = args[4]
        radius = args[2]
        hits: int = np.random.randint(hits_per_rings[0] + self.nof_hits_to_delete,
                                      hits_per_rings[1] + self.nof_hits_to_delete)
        # scale the hits parameter based on the radius: larger radius -> more hits
        hits = int(hits * radius / 5.)

        ellipse = self._init_ellipse(*args[:4], hits)  # type: ignore
        ellipse = ellipse[:hits-self.nof_hits_to_delete]

        while True:
            # remove entries outside of the display and remove duplicates
            ellipse = np.array([x for x in set(tuple(x) for x in ellipse)
                                & set(tuple(x) for x in self.positions)])
            ellipse = np.unique(ellipse, axis=0)

            # check how many hits are shared with other ellipses
            common = np.array([x for x in set(tuple(x) for x in ellipse)
                               & set(tuple(x) for x in self.ring_hits)])
            nof_unique_hits = ellipse.shape[0] - common.shape[0]

            # if the ellipse has at least 8 unique hits, add it to the display,
            # add the hits to the ring_hits array, and break the loop
            if nof_unique_hits >= 8:
                self.ring_hits = np.vstack((self.ring_hits, ellipse))
                break
            # 'ellipse' might be empty if all ellipse hits are outside of the display.
            # In this case, create a new ellipse.
            elif ellipse.shape[0] == 0:
                ellipse = self._init_ellipse(
                    *args[:3], 0.04, 15)  # type: ignore
            else:
                ellipse = np.append(ellipse, self._init_ellipse(
                    *args[:3], 0.04, 15), axis=0)  # type: ignore
        return ellipse

    def add_ellipses(self, nof_rings: int, hits_per_ring: tuple, nof_noise_hits: int = 0) -> None:
        """
        Add a specified number of ellipses with random parameters to the display object.

        The ellipses are added to the display by calling the `_add_ellipse` method. The horizontal and vertical shifts of
        the center of the ellipse, the radius, and the amount of noise are all randomly generated. The number of hits is
        determined by sampling from a uniform distribution between `min_ring_hits` and `max_ring_hits`.

        Parameters:
            nof_rings (int): The number of ellipses to add to the display.
            hits_per_ring (tuple): A tuple of two integers representing the
                minimum and maximum number of hits to add to each ellipse.
                Keep in mind that the number of hits is scaled based on the
                radius of the ellipse in `_add_ellipse`. Thus, hits_per_ring
                does not represent the exact number of hits that will be added
                to each ellipse.

        Returns:
            None
        """
        indices = self._get_indices(nof_rings)
        self.nof_rings = nof_rings
        n = 0
        for n in range(nof_rings):
            # shift the ellipses based on their index
            yshift, xshift = divmod(indices[n], self.shape[1])

            radius: int = np.round(np.random.uniform(3.0, 8.0), 1)
            radius_noise: float = np.random.uniform(0.04, 0.1)

            X = self._add_ellipse(xshift, yshift, radius,
                                  radius_noise, hits_per_ring)

            # add the ellipse to the display by setting the corresponding pixels to 1
            for x, y in zip(X[:, 0], X[:, 1]):
                self[x, y] = 1

            # add 0.5 pixel shift to the center of the ellipse to move it to the
            # center of the pixel
            pars = np.array([xshift+0.5, yshift+0.5, radius, radius, 0])

            # write parameters of each ring into self.params
            self.params[n, :5] = np.array(pars)

        if nof_noise_hits != 0:
            self._add_random_noise(nof_noise_hits)


def delete_files_by_extension(directory, extension):
    for file in tqdm(Path(directory).glob(f"*{extension}")):
        file.unlink()


def add_to_dataset(target_dir: str = 'test', n: int = 100, append: bool = True) -> None:
    """
    Adds event display images and corresponding labels to a dataset.

    This method generates event display images and labels, and adds them to a
    dataset. The event displays are created by adding ellipses with a given
    number of hits, and number of noise hits to a display.  The labels are the
    parameters of the ellipses. The dataset is used to train a convolutional
    neural network to reconstruct the parameters of the ellipses from the event
    display images.

    The event displays have a fixed input shape of (72, 32, 1), and the number
    of rings in the event displays is chosen randomly from the range [0, 4]. The
    number of hits in each ring is chosen randomly from the range [8, 17], and
    the number of noise hits is chosen randomly from the range [0, 9].

    The generated event displays and labels are saved as image files (.png) and
    numpy arrays (.npy), respectively. These files are saved in the specified
    directory.

    Parameters:
        target_dir (str): The directory to save the dataset to.
        n (int): The size of the dataset to create, i.e., the number of event
            display images and labels.
        append (bool): If True, append the generated event displays and labels
            to an existing dataset. If False, delete the existing dataset and create
            a new one.

    Returns:
        None
    """

    input_shape = (72, 32, 1)
    minhits, maxhits = 8, 17
    minrings, maxrings = 0, 4
    min_noise_hits, max_noise_hits = 0, 9

    range_ = range(0)
    if append:
        current_size = len(os.listdir(Path(target_dir, 'X')))
        range_ = range(current_size - 1, current_size - 1 + n)

    else:
        if input('Are you sure you want to delete the existing dataset? (y/n) ') == 'y':
            print(f'Deleting files in {target_dir_X}...')
            delete_files_by_extension(target_dir_X, '.png')

            print(f'Deleting files in {target_dir_y}...')
            delete_files_by_extension(target_dir_y, '.npy')
            range_ = range(n)
        else:
            print('Aborting...')
            sys.exit()

    print(f'Creating images and labels in {target_dir}...')
    for i in tqdm(range_):
        nof_rings = choice(range(minrings, maxrings + 1))
        x = Display(input_shape)
        x.add_ellipses(nof_rings, (minhits, maxhits),
                       choice(range(min_noise_hits, max_noise_hits)))
        y = np.array(x.params)

        im_path = Path(target_dir_X, f'{i}.png')
        label_path = Path(target_dir_y, f'{i}.npy')
        cv2.imwrite(str(im_path), 255*x)
        np.save(str(label_path), y)
    print(f"Done. Created {n} images inside directory '{target_dir}'.")


if __name__ == "__main__":
    """
    This script is used to generate event display images and corresponding
    labels, and adds them to a dataset. The event displays are created by adding
    ellipses with a given number of hits and noise hits to a display. The labels
    are the parameters of the ellipses. The dataset is used to train a
    convolutional neural network to reconstruct the parameters of the ellipses
    from the event display images.

    When run as the main file, it uses argparse to take the target directory and
    number of files as command line arguments. The script takes three parameters
    as input:

        target_dir: The directory to save the dataset to. 
        n_files: The size of the dataset to create, i.e., the number of event
            display images and labels.
        append: If set, append the generated event displays and labels to an
            existing dataset. If not set, delete the existing dataset and create a new
            one. Permission will be asked before deleting the existing dataset.
    For example:
        `python data/create_dataset.py --target_dir=data/train --n_files=1000 --append`
    """
    try:
        __IPYTHON__  # type: ignore
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument('--target_dir', type=str, required=True,
                            help='The directory to save the dataset to.')
        parser.add_argument('--n_files', type=int, required=True,
                            help='The number of files to create.')
        parser.add_argument('--append', action='store_true', required=False,
                            help='''If set, append the data to an existing
                            dataset. If not set, delete the existing dataset and
                            create a new one.''')
        parser.add_argument('--silent', action='store_true', required=False,
                            help='''If set, do not visualize sample images of
                            the dataset after generating it.''')
        args = parser.parse_args()

        target_dir = Path(args.target_dir).resolve()
        n_files = args.n_files
        append = args.append
        silent = args.silent
    else:
        target_dir = Path(root_dir, 'data', 'train')
        n_files = 500
        append = True
        silent = False

    target_dir_X = Path(target_dir, 'X')
    target_dir_y = Path(target_dir, 'y')

    # create target directory and subdirectories if they do not exist
    ignore_rule = "*\n!.gitignore"
    for dir in [target_dir_X, target_dir_y]:
        gitignore_path = Path(dir, '.gitignore')
        path = Path(gitignore_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        path.write_text(ignore_rule)

    add_to_dataset(target_dir=target_dir, n=n_files, append=append)

    # load sample images and labels to verify correctness of the dataset
    datagen = DataGen(target_dir, batch_size=10)
    X, Y = datagen[0]

    # display_images(X)
    fit_rings(X, Y, title=f'Sample images', silent=silent)
