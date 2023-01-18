import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
import tensorflow as tf
import argparse
from utils.utils import *
from models.model import custom_loss
from data.create_data import DataGen


try:
    __IPYTHON__  # type: ignore
except NameError:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='''Path to the model that will be converted to
                        ONNX format.''')
    parser.add_argument('--n_plots', type=int, default=200,
                        help='''Number of plots to be generated.''')
    args = parser.parse_args()
    model_path = os.path.abspath(args.model_path)
    n_plots = args.n_plots
else:
    model_path = 'models/checkpoints/' + '11M-202301152043.model'
    n_plots = 50

n_worst = 50
batch_size = 5000
print(f'Loading model {model_path}...')
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'custom_loss': custom_loss})

print(f'Loading data from data/test...')
dg = DataGen('data/test', batch_size=batch_size)
X, y = dg[0]

y_df = np.array([dg[i][1] for i in range(10)])

ring_params_hist(y_df, title='Test data histograms')

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

fit_rings(X[worst_ids], y_pred[worst_ids], title='worst predictions')

# _______________________________________________________________________________
# Simulation data
# _______________________________________________________________________________

print(f'\nLoading simulation data...')
sim, idealhough, hough = (x[:n_plots] for x in load_sim_data())
print(f'\nPredicting on simulation data...')
y_pred_sim, pred_t = predict(model, sim)

print(f'\nPlotting predictions on simulation data...')
fit_rings(sim, y_pred_sim, title='predictions on simulation data')

ring_params_hist(y_pred_sim, title='Simulation data predictions histograms')
ring_params_hist(idealhough, title='Ideal Hough Transform histograms')
ring_params_hist(hough, title='Hough Transform histograms')
