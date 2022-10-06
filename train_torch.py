import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

from models.model_torch import *
from data.data_torch import *

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"total loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f}\n")

def train_ae(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        decoded, out = model(X)

        loss1 = loss_fn[0](decoded, X)
        loss2 = loss_fn[1](out, y)
        loss = loss1 + loss2

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"total loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_ae(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f}\n")



model = CNN().to(device)
#model = Autoencoder().to(device)
summary(model, (1, 72, 32))

batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)

#%%
epochs = 100
loss_fn = [nn.MSELoss(), nn.BCELoss()]

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epochs)

for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train(train_loader, model, loss_fn[0], optimizer)
    test(test_loader, model, loss_fn[0])
print("Done!")
