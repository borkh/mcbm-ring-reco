import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

root_dir = Path(__file__).parent.parent


class CosineAnnealer:
    """
    A cosine annealer that interpolates between a start and end value over a specified number of steps.

    Args:
        start: The start value.
        end: The end value.
        steps: The number of steps over which to interpolate.
    """

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        """
        Advance the annealer by one step and return the current value.

        Returns:
            The current value of the annealer.
        """
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2.0 * cos


class OneCycleSchedule(tf.keras.callbacks.Callback):
    """
    A learning rate and momentum schedule that follows the "1cycle" schedule, as
    introduced in the paper "Super-Convergence: Very Fast Training of Neural
    Networks Using Large Learning Rates" by Leslie N. Smith
    (https://arxiv.org/abs/1708.07120).

    The schedule consists of two phases:
        1. Increase the learning rate from a inital value to a maximum value
           over a specified number of steps.
        2. Decrease the learning rate from the maximum value to a final value
           that is even smaller than the initial value over the remaining number
           of steps.
    The momentum follows a similar schedule, with the difference that the
    momentum is decreased from a maximum value to a minimum value in the first
    phase, and then increased from the minimum value to the maximum value in
    the second phase.

    This schedule is based on the idea that a model can be trained more
    efficiently if the learning rate is increased and then decreased again over
    the course of training, rather than only decreasing it over time. By gradually
    increasing the learning rate, the model is able to learn more quickly at the
    beginning of training, and then by gradually decreasing the learning rate,
    the model is able to fine-tune its weights and improve its generalization
    performance.

    Args:
        lr_min (float): The minimum learning rate.
        lr_max (float): The maximum learning rate.
        steps (int): The number of steps over which to interpolate the learning
            rate and momentum.
        mom_min (float, optional): The minimum momentum (defaults to 0.85).
        mom_max (float, optional): The maximum momentum (defaults to 0.95).
        phase0perc (float, optional): The percentage of steps in the first phase
            of the schedule (defaults to 0.3).
    """

    def __init__(self, lr_min, lr_max, steps, mom_min=.85, mom_max=.95, phase0perc=0.3):
        super(OneCycleSchedule, self).__init__()
        lr_final = lr_max / 10000
        phase0steps = int(steps * phase0perc)
        phase1steps = int(steps - phase0steps)

        self.phase0steps, self.phase1steps = phase0steps, phase1steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase0steps), CosineAnnealer(mom_max, mom_min, phase0steps)],
                       [CosineAnnealer(lr_max, lr_final, phase1steps), CosineAnnealer(mom_min, mom_max, phase1steps)]]

        # Save the learning rate and momentum values for plotting
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of the training process. Sets the initial
        learning rate and momentum.
        """
        self.phase = 0
        self.step = 0

        lr = self.phases[self.phase][0].start
        mom = self.phases[self.phase][1].start

        self.model.optimizer.lr = lr  # type: ignore
        self.model.optimizer.momentum = mom  # type: ignore

        self.lrs.append(lr)  # lr is saved as a tensor
        self.moms.append(mom)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each batch in the training process.
        Updates the learning rate and momentum.
        """
        lr = self.phases[self.phase][0].step()
        mom = self.phases[self.phase][1].step()

        self.model.optimizer.lr = lr  # type: ignore
        self.model.optimizer.momentum = mom  # type: ignore

        self.lrs.append(lr)  # lr is saved as a tensor
        self.moms.append(mom)
        logs.update({'lr': lr, 'mom': mom})  # type: ignore

        self.step += 1
        if self.step >= self.phase0steps:
            self.phase = 1

    def plot(self):
        """
        Plots the learning rate and momentum schedules over the
        course of the training process.
        """
        figures = [
            px.line(x=np.arange(len(self.lrs)), y=self.lrs),
            px.line(x=np.arange(len(self.moms)), y=self.moms)
        ]
        fig = make_subplots(rows=1, cols=len(figures))
        for i, figure in enumerate(figures):
            for trace in range(len(figure["data"])):  # type: ignore
                fig.append_trace(figure["data"][trace], row=1, col=i+1)
        fig.show()

        # save the plot
        plot_path = str(Path(root_dir, 'plots', 'one_cycle.png'))
        pio.write_image(fig, plot_path)
