from utils.utils import *
from pathlib import Path
import cv2
import pandas as pd
import torchvision
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger  # type: ignore
from pytorch_lightning.profilers import SimpleProfiler
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
    def __init__(self, batch_size, dataset_sizes=None):
        super(LitResNet18, self).__init__()
        # define hyperparameters
        # learning rate is only set for tracking purposes
        # it will be overwritten by the scheduler
        self.learning_rate = 0.1
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes

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
        # log hyperparameters
        self.max_lr = self.learning_rate * 25
        self.save_hyperparameters({
            'batch_size': self.batch_size,
            'initial_lr': self.learning_rate,
            'max_lr': self.max_lr,
            'dataset_sizes': self.dataset_sizes
        })
        # define datasets
        if self.dataset_sizes is None:
            trainsize, valsize, testsize = None, None, None
        else:
            trainsize = self.dataset_sizes[0]
            valsize = self.dataset_sizes[1]
            testsize = self.dataset_sizes[2]
        self.trainset = EventDataset(ROOT_DIR / 'data' / 'train',
                                     n_samples=trainsize,
                                     transforms=transforms.ToTensor())
        self.valset = EventDataset(ROOT_DIR / 'data' / 'val',
                                   n_samples=valsize,
                                   transforms=transforms.ToTensor())

        # define testsets
        self.dataset_names = ['test', 'sim_data']
        self.testset_1 = EventDataset(ROOT_DIR / 'data' / self.dataset_names[0],
                                      n_samples=testsize,
                                      transforms=transforms.ToTensor())
        self.testset_2 = EventDataset(ROOT_DIR / 'data' / self.dataset_names[1],
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

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        # log samples of the model inference
        samples = [plot_single_event(img, pars)
                   for img, pars in zip(x[:10], y_hat[:10])]
        self.logger.log_image(  # type: ignore
            key=f'Model inference on dataset: {self.dataset_names[dataloader_idx]}', images=samples)

        # log the 5 worst predictions per batch
        losses = [self.loss(y_hat[i], y[i]) for i in range(len(y))]
        losses, indices = torch.sort(torch.stack(losses), descending=True)

        worst = x[indices[:5]], y[indices[:5]], y_hat[indices[:5]]

        self.log('test_loss', loss)
        return {'test_loss': loss, 'x': worst[0], 'y': worst[1], 'y_hat': worst[2]}

    def test_epoch_end(self, outputs):
        for idx, name in enumerate(self.dataset_names):
            output = outputs[idx]
            x = torch.cat([out['x'] for out in output])  # type: ignore
            y = torch.cat([out['y'] for out in output])  # type: ignore
            y_hat = torch.cat([out['y_hat'] for out in output])  # type: ignore

            losses = [self.loss(y_hat[i], y[i]) for i in range(len(y))]
            losses, indices = torch.sort(torch.stack(losses), descending=True)

            # log the worst predictions of the whole test set
            n_items = max(50, len(x))
            worst_samples = [plot_single_event(img, pars)
                             for img, pars in zip(x[:n_items], y_hat[:n_items])]
            self.logger.log_image(  # type: ignore
                key=f'Worst predictions on dataset: {name}', images=worst_samples)

            # create histograms of the parameters
            y = y.cpu().numpy().reshape(-1, 25)
            y_hat = y_hat.cpu().numpy().reshape(-1, 25)
            df_y = pd.DataFrame(y).hist(bins=50)
            df_y_hat = pd.DataFrame(y_hat).hist(bins=50)
            print(df_y.shape)
            self.logger.log_image(key=f'Ring parameters of dataset: {name}',  # type: ignore
                                  images=[df_y])
            self.logger.log_image(key=f'Predicted ring parameters of dataset: {name}',  # type: ignore
                                    images=[df_y_hat])
            # self.logger.log_text(key=f'Ring parameters of dataset: {name}',  # type: ignore
            #                      dataframe=df_y)
            # self.logger.log_text(key=f'Predicted ring parameters of dataset: {name}',  # type: ignore
            #                      dataframe=df_y_hat)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          num_workers=12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size,
                          num_workers=12, shuffle=False)

    def test_dataloader(self):
        loader_1 = DataLoader(self.testset_1, batch_size=self.batch_size,  # type: ignore
                              num_workers=12, shuffle=False)
        loader_2 = DataLoader(self.testset_2, batch_size=self.batch_size,  # type: ignore
                              num_workers=12, shuffle=False)
        return [loader_1, loader_2]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            nesterov=True,
            lr=self.learning_rate,
            momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=self.max_lr,
            three_phase=True,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


def train():
    # define model
    model = LitResNet18(batch_size=batch_size, dataset_sizes=dataset_sizes)
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
    model.to_onnx(ROOT_DIR / 'models' / 'lightning-ring-finder.onnx',
                  input_sample=torch.randn(10, 1, 72, 32))


if __name__ == '__main__':
    # define hyperparameters
    batch_size = 1000
    n_epochs = 2
    dataset_sizes = (0, 0, 2000)  # (None, None, None)
    # define logger
    wandb_logger = WandbLogger(project='lightning-ring-finder')
    # simple_profiler = SimpleProfiler()

    # train()

    # load model
    model = LitResNet18.load_from_checkpoint(
        'lightning-ring-finder/hcuvyxnp/checkpoints/epoch=2-step=3000.ckpt',
        batch_size=batch_size,
        dataset_sizes=dataset_sizes)

    trainer = Trainer(accelerator='gpu', devices=1,
                      max_epochs=n_epochs,
                      logger=wandb_logger,
                      #   profiler=simple_profiler,
                      )

    trainer.test(model)
