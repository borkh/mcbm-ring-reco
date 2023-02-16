# AI based RICH ring finder for the mCBM/CBM experiment

## Introduction

This project was created to find an alternative to the Hough Transform based
RICH ring finder that is already implemented in the CbmRoot framework. For this 
purpose, a convolutional neural network (CNN) was created and trained to
predict ring parameters from the input data, which are images of the mRICH
detector (72x32 pixels). The output of the CNN contains the ring parameters:
ring center (x,y) and radius (r). The model is trained using the TensorFlow
Keras API.

<!-- ## File Description

* `data/create_data.py`: Creates training, testing and validation data from
  a toymodel. This data includes images that look like events on the mRICH
  detector with up to five rings as well as the labels --- i.e. ring parameters
  --- for these images. The data is saved in the specified directory (e.g.
  --`data/train`). This directory contains two subdirectories: `X` and `y`. The
   `X` directory contains the images and the `y` directory contains the labels.
  The images are stored in `.png` format and the ring parameters are stored in
  5x5 numpy arrays. Each row of the array contains the ring parameters for one
  ring in the following format:
  ``` [center_x, center_y, semi major axis, semi minor axis, angle] ```
  For now, only perfect rings (not ellipses) are created -> semi major axis and
  semi minor axis are equal and the angle is zero. E.g.:
  ``` [23, 42, 4, 4, 0] ```
* `models/model.py`: Creates the CNN model used for regression of the ring
  parameters. The model is saved in the `models` directory. The model takes in
  images of shape (72,32,1) and outputs numpy arrays of shape (5,5).
* `train.py`: This script trains the model defined in `models/model.py` using
  the data created by `data/create_data.py`. The model is trained using the SGD
  with nestrov momentum with a learning rate schedule based on the `1cycle`
  policy introduced by Leslie N. Smith in his paper "Super-Convergence: Very
  Fast Training of Neural Networks Using Large Learning Rates"
  (https://arxiv.org/abs/1708.07120). The model as well as the training history
  are saved in the `models/checkpoints` directory.
* `utils/one_cycle.py`: Contains the `OneCycleSchedule` class, which implements
  the `1cycle` policy.
* `utils/utils.py`: Contains a variety of utility functions used in the
  project, including functions for visualizing data, fitting ellipses to the
  images, creating histograms and measuring the execution time of other
  functions.

## Installation

To get started, install Anaconda/Miniconda, git and the appropriate versions of
CUDA and cuDNN, if you have a GPU that is compatible with TensorFlow. For more
details, refer to the TensorFlow documentation:
https://www.tensorflow.org/install/gpu

1. Clone the repository
   ```
   git clone https://github.com/borkh/mcbm-ring-reco
   cd mcbm-ring-reco
   ```

2. Create a new conda environment and install the required packages
   ```
   conda env create -f environment.yml
   ```

## Usage

1. Before training the model, the training, testing and validation data needs to
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
   the data is visualized after creation. Set the `--silent` flag to only save
   the plots without showing them.

2. Now, the model can be trained by running:
   ```
   python train.py
   ``` 
   By default, the directories containing the training and validation data are
   set to `data/train` and `data/val`. These can be changed by specifying the
   `--target_dir` and `--val_dir` arguments. For example,
   ```
   python train.py --target_dir=<TARGET_DIR> --val_dir=<VAL_DIR>
   ```
   The model can also be trained in an IPython shell and the arguments can be
   changed manually. You can also run the `LR range test` first by running:
   ```
   python train.py --target_dir=<TARGET_DIR> --find_lr_range
   ```
   This 'LR range test' is described in Leslie N.  Smith's paper "Cyclical
   Learning Rates for Training Neural Networks"
   (https://arxiv.org/abs/1506.01186). This will create a plot from which the
   optimal initial and maximum learning rates for the `1cycle` policy can be
   determined. These values and other hyperparameters can be changed in
   `models/model.py`. The trained models will be saved in the
   `models/checkpoints` directory. After training, predictions on the validation
   data will be made and the results will be plotted as ring fits on the images.
   Set the `--silent` flag to only save the plots without showing them.

3. To evaluate the trained model, run the following command:
   ```
   python evaluate.py --model_path=<MODEL_PATH> --n_plots=<N_PLOTS>
   ``` 
   This will load the model specified by `<MODEL_PATH>`. To evaluate the model,
   the loss and average prediction time will be calculated on the data in
   `data/test`. Furthermore, 50 of the worst predictions will be calculated and
   plotted. Additionally, the `data/sim_data` directory contains simulated data
   from the mRICH detector. That data was created using the CbmRoot framework.
   These images will also be fitted and the first <N_PLOTS> results will be
   plotted.

   Set the `--silent` flag to only save the plots without showing them.
   If the `--model_path` argument is not specified, the latest model in the
   `models/checkpoints` directory will be used.

4. To convert the trained model to ONNX format, run the following command:
   ```
   python onnx/convert_tf2onnx.py --model_path=<MODEL_PATH> --test_dir=<TEST_DIR>
   ``` 
   This will generate a file called `model.onnx` in the `models` directory. This
   file can then be used in the CbmRoot framework for inference. In addition,
   the inference times for both the TensorFlow and ONNX models will be printed
   to the console. Both models will be used to predict the ring parameters for
   the images in the specified test directory (`data/test` by default).
   To verify that the conversion was successful, the predictions of the
   TensorFlow and ONNX models will be compared and the relative and absolute
   differences between them will be printed to the console to ensure they are
   within an acceptable range. Can also be run in an IPython shell. Note, that
   for running the tests after conversion, ROOT needs to be installed.

   If the `--model_path` argument is not specified, the latest model in the
   `models/checkpoints` directory will be used. -->