import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import argparse
import pandas as pd
from tqdm import tqdm

from utils.utils import *
from models.model import custom_loss
from data.create_data import DataGen
from pathlib import Path
from sklearn.metrics import mean_squared_error as mse

root_dir = Path(__file__).parent
test_dir = Path(root_dir, 'data', 'test')

"""
This script evaluates the model on the test data and generates plots of the
histograms of the test data and the model predictions. The worst predictions
are also plotted sorted from best of the worst to worst of the worst. The
plots are saved to the plots directory.

Arguments
---------
    model_path : str
        Path to the model that will be converted to ONNX format. If not
        specified, the most recent model will be used.
    n_plots : int
        Number of plots to be generated. Default is 200.
    silent : bool
        If set, plots will not be shown, but saved to the plots directory.
        Default is False.
"""

try:
    __IPYTHON__  # type: ignore
except NameError:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                                    key=os.path.getctime),
                        help='''Path to the model that will be converted to
                        ONNX format.''')
    parser.add_argument('--n_plots', type=int, default=200,
                        help='''Number of plots to be generated.''')
    parser.add_argument('--silent', action='store_true',
                        help='''If set, plots will not be shown, but saved to
                        the plots directory.''')
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve()
    n_plots = args.n_plots
    silent = args.silent
else:
    # locate the most recent model
    model_path = max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                     key=os.path.getctime)
    n_plots = 200
    silent = True

n_worst = 200
hist_size = 10000
test_size = len(list(Path(test_dir, 'y').glob('*.npy')))
batch_size = test_size // n_worst
plot_dir = root_dir / 'plots' / model_path.stem


print(f'Loading model {model_path}...')
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'custom_loss': custom_loss})
dg = DataGen(test_dir, batch_size=batch_size)

# _______________________________________________________________________________
# Creating histograms of test data
# _______________________________________________________________________________

print(f'\nPlotting test data histograms of size {hist_size}...')
y_df = np.array([dg[j][1] for j in range(hist_size//batch_size)])
ring_params_hist(y_df, plot_dir, title='Test data histograms', silent=silent)


# _______________________________________________________________________________
# Model evaluation
# _______________________________________________________________________________

print(f'\nEvaluating model on {dg.n} events from test data...')
eval_t = evaluate(model, dg)[1]
print(f'Evaluation of {dg.n} took {eval_t}s to run. {eval_t/dg.n}s per event')

# _______________________________________________________________________________
# Selection of worst predictions
# _______________________________________________________________________________

print(f'''\nSplitting test data into {n_worst} batches of size {batch_size}
and selecting the worst predictions for each batch...''')
X_worst, y_worst = [], []
for i in tqdm(range(n_worst), total=n_worst, desc='Batch',
              bar_format='{desc} {n_fmt}/{total_fmt} {percentage:3.0f}%|{bar}|'):
    X, y = dg[i]

    y_pred = model.predict(X, verbose=0)  # type: ignore

    # select the 5 worst predictions for each batch
    diff = np.sum(np.abs(y - y_pred), axis=(1, 2))
    worst_ids = np.argsort(diff)[-5:]
    X_worst.append(X[worst_ids])
    y_worst.append(y[worst_ids])

X_worst = np.concatenate(X_worst)
y_worst = np.concatenate(y_worst)

# Compute worst predictions again to get the worst 50
print(f'\nSelecting the worst {n_worst} predictions...')
y_worst_pred = model.predict(X_worst)  # type: ignore
diff = np.sum(np.abs(y_worst - y_worst_pred), axis=(1, 2))
worst_ids = np.argsort(diff)[-n_worst:]

fit_rings(X_worst[worst_ids], y_worst_pred[worst_ids], plot_dir,
          title='Worst predictions', silent=silent)

# _______________________________________________________________________________
# Simulation data
# _______________________________________________________________________________

print(f'\nLoading simulation data...')
sim, df = (x[:n_plots] for x in load_sim_data())
hough = df.iloc[:, :25].to_numpy().reshape(-1, 5, 5)  # type: ignore
idealhough = df.iloc[:, 25:].to_numpy().reshape(-1, 5, 5)  # type: ignore

print(f'\nPredicting on simulation data...')
y_pred_sim, pred_t = predict(model, sim)

print(f'\nPlotting predictions on simulation data...')
fit_rings(sim, y_pred_sim, plot_dir,
          title='Simulation data inference', silent=silent)
ring_params_hist(y_pred_sim, plot_dir,
                 title='Simulation data inference', silent=silent)

# ring_params_hist(idealhough, plot_dir, title='Ideal HT', silent=silent)
# ring_params_hist(hough, plot_dir, title='HT', silent=silent)

ideal_x_cols = [col for col in df.columns if 'ideal_x' in col]  # type: ignore
ideal_y_cols = [col for col in df.columns if 'ideal_y' in col]  # type: ignore
x_cols = [col for col in df.columns if 'x' in col][:5]  # type: ignore
y_cols = [col for col in df.columns if 'y' in col][:5]  # type: ignore
ideal_r_cols = [col for col in df.columns if 'ideal_a' in col]  # type: ignore
r_cols = [col for col in df.columns if 'a' in col][:5]  # type: ignore


mse_x = np.sqrt(mse(df[ideal_x_cols], df[x_cols]))  # type: ignore
mse_y = np.sqrt(mse(df[ideal_y_cols], df[y_cols]))  # type: ignore
mse_r = np.sqrt(mse(df[ideal_r_cols], df[r_cols]))  # type: ignore
print(f'MSE iHT-HT: \n\tx: {mse_x}, \n\ty: {mse_y}, \n\tr: {mse_r}')

# locate most recent model in models/checkpoints
model_dir = root_dir / 'models' / 'checkpoints'
model_path = max(model_dir.glob('*.model'), key=os.path.getctime)
print(f'Loading model from {model_path}')
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'custom_loss': custom_loss})
pred = model.predict(sim)  # type: ignore

mse_x = np.sqrt(mse(df[ideal_x_cols], pred[..., 0]))  # type: ignore
mse_y = np.sqrt(mse(df[ideal_y_cols], pred[..., 1]))  # type: ignore
mse_r = np.sqrt(mse(df[ideal_r_cols], pred[..., 2]))  # type: ignore

print(f'MSE iHT-CNN: \n\tx: {mse_x}, \n\ty: {mse_y}, \n\tr: {mse_r}')
