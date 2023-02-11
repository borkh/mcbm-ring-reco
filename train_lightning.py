from utils.utils import *
from pathlib import Path
import cv2
from torchvision import transforms, models
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = Path(__file__).parent


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


class LitResNet18(pl.LightningModule):
    def __init__(self, num_classes=5*5, batch_size=200, train_size=None):
        super(LitResNet18, self).__init__()
        self.max_lr = 0.1
        self.batch_size = batch_size
        self.train_size = train_size

        self.resnet = models.resnet18(weights=None)
        # change the input channel to 1
        self.resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.resnet.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 5, 5)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def train_dataloader(self):
        trainset = EventDataset(ROOT_DIR / 'data' / 'train',
                                n_samples=self.train_size, transforms=transforms.ToTensor())
        trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                 num_workers=12, shuffle=True)
        return trainloader

    def val_dataloader(self):
        val_size = int(self.train_size*0.1) if self.train_size is not None else None
        valset = EventDataset(ROOT_DIR / 'data' / 'val',
                              n_samples=val_size, transforms=transforms.ToTensor())  # type: ignore
        valloader = DataLoader(valset, batch_size=self.batch_size,
                               num_workers=12, shuffle=False)
        return valloader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss']  # type: ignore
                               for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            nesterov=True,
            lr=10,
            momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=self.max_lr,
            three_phase=True,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


if __name__ == '__main__':
    # define hyperparameters
    batch_size = 200
    n_epochs = 3
    train_size = None  # 400000

    # define model
    if train_size is None:
        model = LitResNet18(batch_size=batch_size)
    else:
        model = LitResNet18(batch_size=batch_size, train_size=train_size)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(accelerator='gpu',
                      devices=1,
                      max_epochs=n_epochs,
                      auto_lr_find=True,
                      callbacks=[lr_monitor])

    lr_finder = trainer.tuner.lr_find(model)
    init_lr = lr_finder.suggestion()  # type: ignore
    model.max_lr = init_lr * 25  # type: ignore

    trainer.fit(model)
