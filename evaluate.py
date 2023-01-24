import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import argparse
from tqdm import tqdm

from utils.utils import *
from models.model import custom_loss
from data.create_data import DataGen
from pathlib import Path

root_dir = Path(__file__).parent
test_dir = Path(root_dir, 'data', 'test')

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
    model_path = str(Path(args.model_path).resolve())
    n_plots = args.n_plots
    silent = args.silent
else:
    # locate the most recent model
    model_path = max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                     key=os.path.getctime)
    n_plots = 200
    silent = False

n_worst = 50
hist_size = 100000
test_size = len(list(Path(test_dir, 'y').glob('*.npy')))
batch_size = test_size // n_worst


print(f'Loading model {model_path}...')
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'custom_loss': custom_loss})
dg = DataGen(test_dir, batch_size=batch_size)

# _______________________________________________________________________________
# Creating histograms of test data
# _______________________________________________________________________________

print(f'\nPlotting test data histograms of size {hist_size}...')
y_df = np.array([dg[j][1] for j in range(hist_size//batch_size)])
ring_params_hist(y_df, title='Test data histograms', silent=silent)


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

fit_rings(X_worst[worst_ids], y_worst_pred[worst_ids],
          title='Worst predictions', silent=silent)

# _______________________________________________________________________________
# Simulation data
# _______________________________________________________________________________

print(f'\nLoading simulation data...')
sim, idealhough, hough = (x[:n_plots] for x in load_sim_data())
print(f'\nPredicting on simulation data...')
y_pred_sim, pred_t = predict(model, sim)

print(f'\nPlotting predictions on simulation data...')
fit_rings(sim, y_pred_sim, title='Simulation data inference', silent=silent)
ring_params_hist(y_pred_sim, title='Simulation data inference', silent=silent)

# # ring_params_hist(idealhough, title='Ideal HT', silent=silent)
# # ring_params_hist(hough, title='HT', silent=silent)
