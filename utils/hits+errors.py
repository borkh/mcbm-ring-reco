#!/usr/bin/env python3
from utils import *
import sys
sys.path.append('../')

matplotlib.rc('font', **{'size': 30})
plt.rc('lines', linewidth = 4)

mse = tf.keras.losses.MeanSquaredError()

with open('data/sim+idealhough+hough+cnn_autoencoder.pkl', 'rb') as f:
    sim, idealhough, hough, cnn = pkl.load(f)
sim = np.array([cv2.merge((a,a,a)) for a in sim])

print(f'''\n\nMSE\'s of cnn and idealhough: \t x: {mse(cnn[:,:,0],idealhough[:,:,0]).numpy()}
                             y: {mse(cnn[:,:,1],idealhough[:,:,1]).numpy()}
                            r1: {mse(cnn[:,:,2],idealhough[:,:,2]).numpy()}
                            r2: {mse(cnn[:,:,3],idealhough[:,:,3]).numpy()}''')

print(f'''\n\nMSE\'s of idealhough and hough: \t x: {mse(idealhough[:,:,0],hough[:,:,0]).numpy()}
                             y: {mse(idealhough[:,:,1],hough[:,:,1]).numpy()}
                            r1: {mse(idealhough[:,:,2],hough[:,:,2]).numpy()}
                            r2: {mse(idealhough[:,:,3],hough[:,:,3]).numpy()}''')

print(f'''\n\nMSE\'s of cnn and hough: \t x: {mse(cnn[:,:,0],hough[:,:,0]).numpy()}
                             y: {mse(cnn[:,:,1],hough[:,:,1]).numpy()}
                            r1: {mse(cnn[:,:,2],hough[:,:,2]).numpy()}
                            r2: {mse(cnn[:,:,3],hough[:,:,3]).numpy()}''')


# %%
# plot histograms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n_rings_cnn = []
for y in cnn:
    n = len(np.where(np.invert(np.all(y == 0., axis=1)))[0])
    if n <= 4:
        n_rings_cnn.append(n)

n_rings_idealhough = []
for y in idealhough:
    n = len(np.where(np.invert(np.all(y == 0., axis=1)))[0])
    if n <= 4:
        n_rings_idealhough.append(n)

n_rings_hough = []
for y in hough:
    n = len(np.where(np.invert(np.all(y == 0., axis=1)))[0])
    if n <= 4:
        n_rings_hough.append(n)

plt.hist(n_rings_cnn, bins=range(6), align='left', histtype='step', linewidth=6)
plt.hist(n_rings_idealhough, bins=range(6), align='left', histtype='step', linewidth=6)
plt.hist(n_rings_hough, bins=range(6), align='left', histtype='step', linewidth=6)
plt.ylabel('count')
plt.xlabel('number of rings per event')
plt.show()

fig, ax = plt.subplots(1,3)
ax[0].hist(n_rings_cnn, bins=4)
ax[0].set_ylabel('number of rings per event')
ax[0].set_title('regression CNN')

ax[1].hist(n_rings_idealhough, bins=4)
ax[1].set_ylabel('number of rings per event')
ax[1].set_title('ideal HTM')

ax[2].hist(n_rings_hough, bins=4)
ax[2].set_ylabel('number of rings per event')
ax[2].set_title('HTM')
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
