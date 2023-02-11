import torch
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import transforms, models
from torch.nn.functional import relu
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, MaxPool2d, Linear,
                      Flatten, ReLU, Sequential)

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import plotly.express as px

from utils.utils import *

root_dir = Path(__file__).parent


class EventDataset(Dataset):

    def __init__(self, target_dir, transforms=None, n_samples=None):
        super(EventDataset, self).__init__()
        self.target_dir = Path(target_dir).absolute()
        self.transforms = transforms

        if n_samples is None:
            self.n_samples = len(list((self.target_dir / 'y').glob('*.npy')))
        else:
            self.n_samples = n_samples

    def __getitem__(self, index):
        X_dir = self.target_dir / 'X'
        y_dir = self.target_dir / 'y'
        x = cv2.imread(str(X_dir / f'{index}.png'), cv2.IMREAD_GRAYSCALE)
        y = np.load(y_dir / f'{index}.npy').astype(np.float32)

        if self.transforms:
            x = self.transforms(x)
        y = torch.from_numpy(y)

        return x, y

    def __len__(self):
        return self.n_samples


class RingRegressor(torch.nn.Module):
    def __init__(self, nif=8, kernel_size=3):
        super(RingRegressor, self).__init__()
        self.nif = nif
        self.kernel_size = kernel_size

        self.network = torch.nn.Sequential(
            self.conv_block(1, nif),
            self.conv_block(nif, nif * 2),
            self.conv_block(nif * 2, nif * 4),
            self.conv_block(nif * 4, nif * 8, extra_conv=True),
            self.conv_block(nif * 8, nif * 16, extra_conv=True),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 1 * nif * 16, 25),
        )

    def conv_block(self, in_channels, out_channels, extra_conv=False):
        layers = [
            torch.nn.Conv2d(in_channels, out_channels,
                            self.kernel_size, padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels,
                            self.kernel_size, padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        ]
        if extra_conv:
            layers += [
                torch.nn.Conv2d(out_channels, out_channels,
                                self.kernel_size, padding="same"),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
            ]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out.view(-1, 5, 5)


class ReshapedResNet18(torch.nn.Module):
    def __init__(self, num_classes=5*5):
        super(ReshapedResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)
        # change the input channel to 1
        self.resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc2(x)
        x = x.view(-1, 5, 5)
        return x

        # use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define hyperparameters
batch_size = 200
n_epochs = 3

# define model
# model = RingRegressor().to(device)
model = ReshapedResNet18().to(device)

summary(model, (1, 72, 32))

# define dataset and dataloader
transforms = transforms.ToTensor()
dataset = EventDataset('data/train', transforms=transforms, n_samples=400000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# define loss, optimizer and scheduler
mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# use 1cycle policy
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  # type: ignore
                                                max_lr=0.1,
                                                steps_per_epoch=len(
                                                    dataloader),
                                                epochs=n_epochs)


for epoch in range(n_epochs):
    pbar = tqdm(dataloader)
    for batch, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        loss = mse(y_pred, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if batch % 5 == 0:
            pbar.set_description(
                f'Epoch {epoch+1}/{n_epochs}: Loss: {loss.item():>8.4f}')
