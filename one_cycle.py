#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2.0 * cos

class OneCycleSchedule(tf.keras.callbacks.Callback):
    def __init__(self, lr_min, lr_max, steps, mom_min=.85, mom_max=.95, phase0perc=0.3):
        super(OneCycleSchedule, self).__init__()
        lr_final = lr_min / 1 # TODO: change denominator again
        phase0steps = int(steps * phase0perc)
        phase1steps = int(steps - phase0steps)

        self.phase0steps, self.phase1steps = phase0steps, phase1steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase0steps), CosineAnnealer(mom_max, mom_min, phase0steps)],
                       [CosineAnnealer(lr_max, lr_final, phase1steps), CosineAnnealer(mom_min, mom_max, phase1steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        #        self.plot()
        self.phase = 0
        self.step = 0

        self.set_lr(self.phases[self.phase][0].start)
        self.set_mom(self.phases[self.phase][1].start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_mom())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase0steps:
            self.phase = 1

        self.set_lr(self.phases[self.phase][0].step())
        self.set_mom(self.phases[self.phase][1].step())
        #logs.update({'lr' : self.get_lr(),
        #             'mom': self.get_mom()})

    def on_train_epoch_end(self, epoch, logs=None):
        print("\t", self.get_lr(), self.get_mom())

    def get_lr(self):
        try:
            return K.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_mom(self):
        try:
            return K.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            K.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass

    def set_mom(self, mom):
        try:
            K.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass

    def plot(self):
        """
        lr0 = [self.phases[0][0].step() for _ in range(self.phase0steps)]
        lr1 = [self.phases[1][0].step() for _ in range(self.phase1steps)]
        lr = np.array(lr0 + lr1)

        mom0 = [self.phases[0][1].step() for _ in range(self.phase0steps)]
        mom1 = [self.phases[1][1].step() for _ in range(self.phase1steps)]
        mom = np.array(mom0 + mom1)
        """
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
        plt.savefig("plots/1cycle.png")
