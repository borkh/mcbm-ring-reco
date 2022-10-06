#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import cv2, matplotlib
from mpl_toolkits.axisartist.axislines import SubplotZero


font = {'family' : 'normal',
        'size'   : 30}

matplotlib.rc('font', **font)
plt.rc('lines', linewidth=6)

x = np.linspace(-5, 10, 1000)



#img = cv2.imread('/home/robin/over-underfitting.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


fig, ax = plt.subplots(1,1)


ax.plot(x, 0.4*(x-3.5)**2 + 3, label='test loss')
ax.plot(x, np.e**(-0.5*(x-3.5))+1, label='training loss')
ax.vlines(3.5, 0, 15, linestyle='dashdot', color='black', linewidth=3, label='optimal model complexity')

ax.set_xlim(0,7)
ax.set_ylim(0,12)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    ax.spines[side].set_visible(False)

# removing the axis ticks
plt.xticks([]) # labels
plt.yticks([])

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 0.2
hl = 0.2
#hw = 1./20.*(ymax-ymin)
#hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.0 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl*0.6, overhang = ohg,
         length_includes_head=True, clip_on=False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw*0.6, head_length=yhl, overhang = ohg,
         length_includes_head=True, clip_on=False)

#ax.imshow(img)

#ax.set_xticks([])
#ax.set_yticks([])
ax.set_xlabel('model complexity')
ax.set_ylabel('loss')
ax.legend()

plt.show()
