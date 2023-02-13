from utils.utils import *
from pathlib import Path
import cv2
import torchvision
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger  # type: ignore
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
    def __init__(self, batch_size, train_size=None):
        super(LitResNet18, self).__init__()
        # define hyperparameters
        # learning rate is only set for tracking purposes
        # it will be overwritten by the scheduler
        self.learning_rate = 0.1
        self.batch_size = batch_size
        self.train_size = train_size

        # define loss
        self.loss = torch.nn.MSELoss()

        # define model
        self.resnet = models.resnet18(weights=None)
        # change the input channel to 1
        self.resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = torch.nn.Linear(512, 25)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 5, 5)
        return x

    def setup(self, stage=None):
        self.trainset = EventDataset(ROOT_DIR / 'data' / 'train',
                                     n_samples=self.train_size, transforms=transforms.ToTensor())
        self.valset = EventDataset(ROOT_DIR / 'data' / 'val',
                                   n_samples=None,
                                   transforms=transforms.ToTensor())
        self.testset = EventDataset(ROOT_DIR / 'data' / 'test',
                                    n_samples=None,
                                    transforms=transforms.ToTensor())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss']  # type: ignore
    #                            for x in outputs]).mean()
    #     self.log('avg_val_loss', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        sample_imgs = x[:5]

        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('images', grid, 0)  # type: ignore

        self.log('test_loss', loss)
        return {'test_loss': loss}

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          num_workers=12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size,
                          num_workers=12, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size,
                          num_workers=12, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            nesterov=True,
            lr=self.learning_rate,
            momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=self.learning_rate * 25,
            three_phase=True,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


def train():
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
                      #   auto_scale_batch_size='binsearch',
                      callbacks=[lr_monitor],
                      logger=wandb_logger)
    trainer.tune(model)

    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    # define hyperparameters
    batch_size = 200
    n_epochs = 3
    train_size = 200000
    # define logger
    wandb_logger = WandbLogger(project='lightning-ring-finder')

    # train()

    # load model
    # model = LitResNet18.load_from_checkpoint(
    #     'lightning_logs/version_34/checkpoints/epoch=2-step=255000.ckpt')
    model = LitResNet18.load_from_checkpoint(
        'lightning-ring-finder/hcuvyxnp/checkpoints/epoch=2-step=3000.ckpt', batch_size=batch_size)

    trainer = Trainer(accelerator='gpu', devices=1,
                      max_epochs=n_epochs, logger=wandb_logger)
    trainer.test(model)
