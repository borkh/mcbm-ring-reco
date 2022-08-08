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

#df = pd.read_csv("wandb_export_2022-05-05T14_45_43.219+02_00.csv", header=0)
#df = pd.read_csv("wandb_export_2022-08-08T10_07_31.637+02_00.csv", header=0)
df = pd.read_csv("wandb_export_2022-08-08T11_24_12.761+02_00.csv", header=0)


step = df["Step"]
#loss = df["warm-sweep-1 - loss"]
#val_loss = df["warm-sweep-1 - val_loss"]
loss = df["pretty-sweep-1 - loss"]
val_loss = df["pretty-sweep-1 - val_loss"]

fig, ax = plt.subplots(1,2)

last_ep = 19
ax[0].plot(step, loss, label="{}: {} loss".format(last_ep, np.round(loss[last_ep],4)))
ax[0].plot(step, val_loss, label="{}: {} val_loss".format(last_ep, np.round(val_loss[last_ep],4)))
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss")
#ax[0].set_ylim(0, 3.5)
ax[0].legend()

#df = pd.read_csv("wandb_export_2022-05-05T15_25_10.582+02_00.csv", header=0)

acc = df["pretty-sweep-1 - acc"]
val_acc = df["pretty-sweep-1 - val_acc"]

ax[1].plot(step, acc, label="{}: {} accuracy".format(last_ep, np.round(acc[last_ep],4)))
ax[1].plot(step, val_acc, label="{}: {} val_accuracy".format(last_ep, np.round(val_acc[last_ep],4)))
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy")
#ax[1].set_ylim(0.86, 0.99)
ax[1].legend()
plt.show()


"""
fig, axis = plt.subplots(2,2)
df = pd.read_csv("run-train-tag-Loss_classification_loss.csv", header=0)
step = df["Step"]
loss = df["Value"]
ax = axis[0,0]
ax.plot(step, loss)
ax.set_ylabel("classification loss")
ax.set_xlabel("steps")

df = pd.read_csv("run-train-tag-Loss_localization_loss.csv", header=0)
step = df["Step"]
loss = df["Value"]
ax = axis[0,1]
ax.plot(step, loss)
ax.set_ylabel("localization loss")
ax.set_xlabel("steps")

df = pd.read_csv("run-train-tag-Loss_regularization_loss.csv", header=0)
step = df["Step"]
loss = df["Value"]
ax = axis[1,0]
ax.plot(step, loss)
ax.set_ylabel("regularization loss")
ax.set_xlabel("steps")

df = pd.read_csv("run-train-tag-Loss_total_loss.csv", header=0)
step = df["Step"]
loss = df["Value"]
ax = axis[1,1]
ax.plot(step, loss)
ax.set_ylabel("total loss")
ax.set_xlabel("steps")

plt.show()


df = pd.read_csv("run-train-tag-learning_rate.csv", header=0)
step = df["Step"]
loss = df["Value"]
plt.plot(step, loss)
plt.ylabel("learning rate")
plt.xlabel("steps")
plt.show()
"""
