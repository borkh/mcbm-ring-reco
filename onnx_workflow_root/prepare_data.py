import numpy as np
import plotly.express as px
import cv2

import sys
np.set_printoptions(threshold=sys.maxsize)

dir_ = '/home/robin/mcbm-ring-reco/data/test/X/'
if_ = dir_ + '1.png'

img = cv2.imread(if_)/255

px.imshow(img).show()

# write img values to csv file
np.savetxt('img.csv', img[:, :, 0], delimiter=',')
