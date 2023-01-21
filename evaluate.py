import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import argparse
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
    parser.add_argument('--model_path', type=str, required=True,
                        help='''Path to the model that will be converted to
                        ONNX format.''')
    parser.add_argument('--n_plots', type=int, default=200,
                        help='''Number of plots to be generated.''')
    parser.add_argument('--silent', action='store_true',
                        help='''If set, plots will not be shown, but saved
                          to the plots directory.''')
    args = parser.parse_args()
    model_path = str(Path(args.model_path).resolve())
    n_plots = args.n_plots
    silent = args.silent
else:
    # locate the most recent model
    model_path = max(list(Path(root_dir, 'models', 'checkpoints').glob('*.model')),
                     key=os.path.getctime)
    n_plots = 50
    silent = False

n_worst = 50
batch_size = 5000
print(f'Loading model {model_path}...')
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'custom_loss': custom_loss})

print(f'Loading data from {test_dir}...')
dg = DataGen(test_dir, batch_size=batch_size)
X, y = dg[0]

y_df = np.array([dg[i][1] for i in range(10)])

ring_params_hist(y_df, title='Test data histograms', silent=True)

print(f'\nEvaluating model on {dg.n} events from test data...')
eval_t = evaluate(model, dg)[1]  # type: ignore
print(
    f'Evaluation of {dg.n} took {eval_t}s to run. {eval_t / dg.n}s per event')
print(f'\nPredicting on {batch_size} events from test data...')
y_pred, pred_t = predict(model, X)
print(f'Inference took {pred_t}s to run. {pred_t / len(X)}s per event')
print(f'\nSelecting {n_worst} of the worst predictions and plotting them...')
# select worst predictions
diff = np.sum(np.abs(y - y_pred), axis=(1, 2))
worst_ids = np.argsort(diff)[-n_worst:]

fit_rings(X[worst_ids], y_pred[worst_ids],
          title='Worst predictions', silent=silent)

# _______________________________________________________________________________
# Simulation data
# _______________________________________________________________________________

print(f'\nLoading simulation data...')
sim, idealhough, hough = (x[:n_plots] for x in load_sim_data())
print(f'\nPredicting on simulation data...')
y_pred_sim, pred_t = predict(model, sim)

print(f'\nPlotting predictions on simulation data...')
fit_rings(sim, y_pred_sim, title='predictions on simulation data', silent=silent)


ring_params_hist(
    y_pred_sim, title='Simulation data predictions histograms', silent=silent)
ring_params_hist(
    idealhough, title='Ideal Hough Transform histograms', silent=silent)
ring_params_hist(hough, title='Hough Transform histograms', silent=silent)
