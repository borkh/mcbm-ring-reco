import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import numpy as np
import os
import cv2
from PIL import Image
import pickle as pkl

import psutil
import gc
import matplotlib.pyplot as plt

from tqdm import tqdm

class FileDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.transform(self.data[index])
        X = X.to(device, dtype=torch.float)

        y = self.target[index]
        y = torch.from_numpy(y.flatten())
        y = y.to(device, dtype=torch.float)
        return X, y

class ImageDataset(Dataset):
    def __init__(self, dir_, transform=None):
        self.dir = dir_
        self.len = len(os.listdir(f'{dir_}/X'))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_loc = f'{self.dir}/X/{index}.png'
        par_loc = f'{self.dir}/y/{index}.npy'
        #X = cv2.imread(img_loc)
        X = Image.open(img_loc)
        y = np.load(par_loc)

        X = self.transform(X)
        #X, y = torch.from_numpy(X), torch.from_numpy(y)
        #X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
        y = torch.from_numpy(y)
        y = y.to(device, dtype=torch.float)

        return X, y


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#%%
#length = 1000000
length = len(os.listdir('data/X'))
split = int(length * 0.9)
batch_size = 500

transform = transforms.Compose([transforms.ToTensor()])
data = ImageDataset('data/', transform)

#with open('data/100k-fixed1.pkl', 'rb') as f:
#    x, y = pkl.load(f)
#data = FileDataset(x, y, transform)
train_set, valid_set, _ = random_split(data, (split,length-split, 0))

train_dl = DataLoader(train_set, batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size, shuffle=True)

#print(f"RAM memory % used :{psutil.virtual_memory()[2]}")

if __name__ == '__main__':
    # load data
    import pickle as pkl
    size = 100000
    for i in tqdm(range(15)):
        with open(f'data/100k-fixed{i+1}.pkl', 'rb') as f:
            img, par = pkl.load(f)

        par = np.reshape(par, (par.shape[0], 25))

        for j in range(len(img)):
            cv2.imwrite(f'data/X/{i*size + j}.png', 255*img[j])
            np.save(f'data/y/{i*size + j}.npy', par[j])
    print("Done.")
