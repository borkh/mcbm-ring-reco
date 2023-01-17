# AI based RICH ring finder for the mCBM/CBM experiment

## Introduction

This project was created to find an alternative to the Hough Transform based
RICH ring finder that is already implemented in the CbmRoot framework. For this 
purpose, a convolutional neural network (CNN) was created and trained to
predict ring parameters from the input data, which are images of the mRICH
detector (72x32 pixels). The output of the CNN contains the ring parameters:
ring center (x,y) and radius (r). The model is trained using the TensorFlow
Keras API.

## File Description

* `data/create_data.py`: Creates training, testing and validation data from
    a toymodel, which creates images that look like events on the mRICH detector
    with up to five rings. The data is saved in `data/train`, `data/test` and
    `data/val`.  Each of these folders contains two subfolders: X and y. The X
    folder contains the images and the y folder contains the ring parameters.
    The images are stored in `.png` format and the ring parameters are stored in
    5x5 numpy arrays. Each row of the array contains the ring parameters for one
    ring in the following format:
    ``` [center_x, center_y, semi major axis, semi minor axis, angle] ```
    For now, only perfect rings (not ellipses) are created -> semi major axis
    and semi minor axis are equal and the angle is zero. E.g.:
    ``` [42, 23, 4, 4, 0] ```
* `models/model.py`: Creates the CNN model used for regression. The model is
    saved in the `models` directory. The model takes in images of shape
    (72,32,1) and outputs numpy arrays of shape (5,5).
* `train.py`: This script trains the model defined in `models/model.py` using
    the data created by `data/create_data.py` and stored in `data/train`. The
    data in `data/val` is used for validation. The model is trained using the
    SGD with nestrov momentum with a learning rate schedule based on the
    `1cycle` policy introduced by Leslie N. Smith in his paper
    "Super-Convergence: Very Fast Training of Neural Networks Using Large
    Learning Rates" (https://arxiv.org/abs/1708.07120). The model as well as the
    training history are saved in the `models/checkpoints` directory.
* `utils/one_cycle.py`: Contains the `OneCycleSchedule` class, which implements 
    the `1cycle` policy.
* `utils/utils.py`: Contains a variety of utility functions used in the
    project, including functions for visualizing the training data, fitting
    ellipses to the images, and measuring the execution time of other functions.

## Usage

1. Clone the repository
```
git clone https://github.com/borkh/mcbm-ring-reco
```

2. Install the required packages
```
pip install -r requirements.txt
```

3. After first cloning the repository, training and validation data has to be
   created. This can be done by running `data/create_data.py` and specifying the
   target directory as well as the number of files to create. Optionally, the
   `--append` flag can be used to append the data to existing files or to remove
   existing files before creating new ones. The default is to remove existing
   files. By default, the data is also visualized after creation. This can be
   disabled by setting the `--visualize` flag to `False`.  For example, to
   append 1000 files to the `data/train` directory without visualization, run:
   ```
   python data/create_data.py --target=data/train --n_files=1000 --append=True --visualize=False
   ```
    Alternatively, the file can be run in an IPython shell and the arguments can
    be changed manually. Make sure to also populate the `data/test` and
    `data/val` directories with data before continuing.  

4. Now, the model can be trained by running
    ```
    python train.py
    ``` 
    By default, the directories containing the training and validation data are
    set to `data/train` and `data/val`. These can be changed by specifying the
    `--target_dir` and `--val_dir` arguments. For example,
    ```
    python train.py --target_dir=<TARGET_DIR> --val_dir=<VAL_DIR>
    ``` 
    The model can also be trained in an IPython shell and
    the arguments can be changed manually.
    You can also run the `LR range test` first by running
    ```
    python train.py --target_dir=<TARGET_DIR> --find_lr_range
    ```
    This 'LR range test' is described in Leslie N.
    Smith's paper "Cyclical Learning Rates for Training Neural Networks"
    (https://arxiv.org/abs/1506.01186). This will create a plot from which the
    optimal initial and maximum learning rates for the `1cycle` policy can be
    determined. These values and other hyperparameters can be changed in
    `models/model.py`. The trained models will be saved in the
    `models/checkpoints` directory. After training, some predictions on the
    validation data will be made and the results will be plotted.

5. To evaluate the trained model, run the following command:
   ```
   python evaluate.py --model_path=<MODEL_PATH> --nof_plots=<NOF_PLOTS>
   ``` 
   This will load the model specified by `<MODEL_PATH>`. To evaluate the model,
   the loss and average prediction time will be calculated on the data in
   `data/test`. Furthermore, the 50 worst predictions will be calculated and
   plotted. Additionally, `data/sim_data` directory contains simulated data from
   the mRICH detector. That data was created using the CbmRoot framework.  These
   images will also be fitted and the first <NOF_PLOTS> results will be plotted.

6. To convert the trained model to ONNX format, run the following command:
   ```
   python onnx/convert_tf2onnx.py --model_path=<MODEL_PATH>
   ``` 
   This will generate a file called `model.onnx` in the `models` directory. This
   file can then be used in the CbmRoot framework for inference. In addition,
   the inference times for both the TensorFlow and ONNX models will be printed
   to the console. Both models will also be used to predict the ring parameters
   for the images in the `data/test` directory. To verify that the conversion was
   successful, the predictions of the TensorFlow and ONNX models will be
   compared and the relative and absolute differences between them will be
   printed to the console to ensure they are within an acceptable range.
   Can also be run in an IPython shell. Note, that this script requires the
   ROOT framework to be installed.