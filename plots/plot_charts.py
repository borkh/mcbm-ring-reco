#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family' : 'normal',
        'size'   : 32}

matplotlib.rc('font', **font)
matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['axes.linewidth'] = 0

df = pd.read_csv("wandb_export_2022-05-05T14_45_43.219+02_00.csv", header=0)

print(df)
step = df["Step"]
loss = df["warm-sweep-1 - loss"]
val_loss = df["warm-sweep-1 - val_loss"]

last_ep = 23
plt.plot(step, loss, label="{}: {} loss".format(last_ep, np.round(loss[last_ep],4)))
plt.plot(step, val_loss, label="{}: {} val_loss".format(last_ep, np.round(val_loss[last_ep],4)))
plt.xlabel("epoch")
plt.ylabel("mean squared error")
plt.ylim(0, 3.5)
plt.legend()
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.99, top=0.98)
plt.grid()
plt.show()

df = pd.read_csv("wandb_export_2022-05-05T15_25_10.582+02_00.csv", header=0)

print(df)
step = df["Step"]
loss = df["warm-sweep-1 - accuracy"]
val_loss = df["warm-sweep-1 - val_accuracy"]

last_ep = 23
plt.plot(step, loss, label="{}: {} accuracy".format(last_ep, np.round(loss[last_ep],4)))
plt.plot(step, val_loss, label="{}: {} val_accuracy".format(last_ep, np.round(val_loss[last_ep],4)))
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0.86, 0.99)
plt.legend()
plt.subplots_adjust(left=0.083, bottom=0.1, right=0.99, top=0.98)
plt.grid()
plt.show()
