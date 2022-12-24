#!/usr/bin/env python3
from utils import *
import numpy as n
import matplotlib


matplotlib.rc('font', **{'size': 20})
plt.rc('lines', linewidth=4)

with open('../data/sim_data/sim+idealhough+hough+cnn.pkl', 'rb') as f:
    sim, idealhough, hough, cnn1 = pkl.load(f)
sim = np.array([cv2.merge((a, a, a)) for a in sim])

# with open('/home/robin/cbm-ring-reco/images/sim+pred_images+idealhough+hough+cnn.pkl', 'rb') as f:
#    _, _, _, _, cnn2 = pkl.load(f)

# create ring fits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
idealhough_fit = np.array([plot_single_event(x, y1)
                          for x, y1 in zip(sim[:50], idealhough[:50])])
cnn_fit = np.array([plot_single_event(x, y1)
                   for x, y1 in zip(sim[500:1000], cnn1[500:1000])])

cnn_vs_idealhough = np.array([plot_single_event(x, y1, y2) for x, y1, y2 in zip(
    sim[500:1000], cnn1[500:1000], idealhough[500:1000])])
cnn_vs_hough = np.array([plot_single_event(x, y1, y2)
                        for x, y1, y2 in zip(sim[:500], cnn1[:500], hough[:500])])
hough_vs_idealhough = np.array([plot_single_event(
    x, y1, y2) for x, y1, y2 in zip(sim[:50], hough[:50], idealhough[:50])])

display_images(3, 5, cnn_fit, 1, 10)
# display_images(3,5,cnn_vs_idealhough,1,10)
# display_images(3,5,cnn_vs_hough,1,10)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# search bad ring fits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#all_fits = np.array([plot_single_event(x,y1,y2,y3,y4) for x,y1,y2,y3,y4 in zip(sim[500:1000], cnn1[500:1000], cnn2[500:1000], idealhough[500:1000], hough[500:1000])])
# display_images(3,5,all_fits,1,10)
