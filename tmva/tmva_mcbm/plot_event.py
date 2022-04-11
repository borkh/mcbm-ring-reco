#!/usr/bin/env python3
from visual_functions import *

def get_data(rootfile, index):
    f = ROOT.TFile.Open(rootfile)
    tree = f.Get("train")

    tree.GetEntry(index)
    return np.array([*tree.x]).reshape(72,32,1), np.array([*tree.y])

if __name__ == "__main__":
    f = "datasets/test.root"
    imgs = np.array([get_data(f, i)[0] for i in range(100)])
    pars = np.array([get_data(f, i)[1] for i in range(100)])
    for i in range(5):
        #plt.imshow(plot_single_event(imgs[i], pars[i]))
        plt.imshow(imgs[i])
        plt.show()
#    display_data(imgs, 0)
