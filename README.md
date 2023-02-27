# AI based RICH ring finder for the mCBM/CBM experiment

## Introduction

This project was created to find an alternative to the Hough Transform based
RICH ring finder that is already implemented in the CbmRoot framework. For this 
purpose, a convolutional neural network (CNN) was created and trained to
predict ring parameters from the input data, which are images of the mRICH
detector (72x32 pixels). The output of the CNN contains the ring parameters:
ring center (x,y) and radius (r). The model is trained using the PyTorch
Lightning framework.

## File Description

* `data/create_data.py`: Creates training, testing and validation data from
  a toymodel. This data includes images that look like events on the mRICH
  detector with up to five rings as well as the labels - i.e. ring parameters
  - for these images. The data is saved in the specified directory (e.g.
  `data/train`). This directory contains two subdirectories: `X` and `y`. The
   `X` directory contains the images and the `y` directory contains the labels.
  The images are stored in `.png` format and the ring parameters are stored in
  5x5 numpy arrays. Each row of the array contains the ring parameters for one
  ring in the following format:
  ``` [center_x, center_y, semi major axis, semi minor axis, angle] ```
  For now, only perfect rings (not ellipses) are created -> semi major axis and
  semi minor axis are equal and the angle is zero. E.g.:
  ``` [23, 42, 4, 4, 0] ```
* `hyperparameters.yml`: Contains the hyperparameters used for training the
  model.
* `train.py`: This script trains and evaluates a model using the data created by
  `data/create_data.py`. The script defines how the data is loaded with the
  `EventDataset` and `EventDataModule` classes. The model and the
  training/validation/testing loops are defined in the `TorchModel` and 
  `LitModel` classes.
* `utils.py`: Contains a variety of utility functions used in the
  project, including functions for visualizing data, fitting ellipses to the
  images, creating histograms and measuring the execution time of other
  functions.
* `data/sim_data`: Contains two .csv files with simulation data from the mRICH
  detector. The data was created using the CbmRoot framework. 
* `onnx`: Contains scripts to run the ONNX model with the CbmRoot framework.

## Installation

To get started, install Anaconda/Miniconda, git and the appropriate versions of
CUDA and cuDNN, if you have a GPU that is compatible with PyTorch. If you don't
already have a WandB account and you want to log your training runs, you can
create one here: https://wandb.ai/site.  Then, follow the steps below:

1. Clone the repository
   ```
   git clone https://github.com/borkh/mcbm-ring-reco
   cd mcbm-ring-reco
   ```

2. Create a new conda environment and install the required packages
   ```
   conda env create -f environment.yml
   ```

## Creating the datasets
   Before training the model, the training, testing and validation data needs to
   be created. This can be done by running the following command:
   ```
   python data/create_data.py --target_dir=<TARGET_DIR> --n_files=<N_FILES>
   ```
   Alternatively, the file can be run in an IPython shell and the arguments can
   be changed manually. This needs to be run three times, once for each of the
   `data/train`, `data/test` and `data/val` directories. The `--target_dir`
   argument specifies the directory in which the data will be saved. The
   `--n_files` argument specifies the number of files that will be created.
   To automatically create the data for all three directories, run:
   ```
   python data/create_data.py --n_files=<N_FILES> --auto
   ```
   The ratio of the number of files in the training, testing and validation
   directories are set to 0.8, 0.1 and 0.1, respectively.

   Optionally, the `--append` flag can be set to append the data to existing
   files. Otherwise, the existing files will be overwritten. Set `--force` to
   overwrite the existing files without asking for confirmation. By default,
   the data is visualized after creation. Set the `--silent` flag to disable
   this.
   To create a dataset in `data/sim_data` from the CBM simulation data, run:
   ```
   python utils.py
   ```

## Training the model
   For a complete training and evaluation run, run the following command:
   ```
   python train.py
   ``` 
   The hyperparameters used for training are specified in `hyperparameters.yml`.
   The learning rate is not specified in this file, as the `LR range test` is
   used to find the optimal learning rate. The resulting learning rate is then
   used for the SGD optimizer together with the `1cycle` policy. For more
   information on the `LR range test` and `1cycle` policy, see the following
   papers:
   * Leslie N. Smith: Cyclical Learning Rates for Training Neural Networks
     (https://arxiv.org/abs/1506.01186)
   * Leslie N. Smith: Super-Convergence: Very Fast Training of Neural Networks
     Using Large Learning Rates (https://arxiv.org/abs/1708.07120)

   The model is trained using the PyTorch Lightning framework. The model is
   saved in the `models`, where a new directory is created for each run with the
   naming convention `version_X`, where `X` is the version number. After
   training, is converted to ONNX format and saved in the same directory.
   Additionally, he last model checkpoint is logged to WandB.
   
   All the hyperparameters, the learning rate schedule as well as the
   train/val/test losses are logged.  Furthermore, after training, the model is
   evaluated on two test sets: one from the toymodel and one from the real data.
   Some sample images with predicted ring fits are also logged. Additionally,
   for each test batch the worst predictions are logged, along with the
   corresponding DataFrames and Histograms of those predictions.
   
   If you only want to evaluate an existing model, run:
   ```
   python train.py --eval --version=<VERSION>
   ```
   Where `<VERSION>` is the version number of the model you want to evaluate, e.g.
   `12` for `version_12`. The model will be loaded from the `models` directory.