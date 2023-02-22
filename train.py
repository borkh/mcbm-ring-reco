import argparse
import sys
from pathlib import Path
from typing import Union, Optional

import cv2
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import wandb
from utils import *


ROOT_DIR = Path(__file__).parent
WANDB_DIR = ROOT_DIR / 'wandb'
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'

NAME = 'version_'
versions = [int(i.name.split(NAME)[1])
            for i in WANDB_DIR.glob(f'*{NAME}*')]
if versions == []:
    VERSION = 0
else:
    VERSION = max(versions) + 1


class EventDataset(Dataset):
    """
    Dataset that loads images and ring parameters from a directory.
    The directory structure should be as follows:
    target_dir
    ├── X
    │   ├── 0.png
    │   ├── 1.png
    │   ├── ...
    ├── y
    │   ├── 0.npy
    │   ├── 1.npy
    │   ├── ...

    Args
    ----
        target_dir: Path
            Path to the directory containing the images and ring parameters.
        transforms: torchvision.transforms
            Transforms to apply to the images.
        n_samples: int
            Number of samples to load from the directories. If None, all
            samples are loaded.
    """

    def __init__(self, target_dir: Union[str, Path],
                 transforms: Optional[transforms.Compose] = None,
                 n_samples: Optional[int] = None):
        super(EventDataset, self).__init__()
        self.target_dir = Path(target_dir).absolute()
        self.transforms = transforms

        # if n_samples is not specified, load all samples from the directory
        if n_samples is None:
            self.n_samples = len(list((self.target_dir / 'y').glob('*.npy')))
        else:
            self.n_samples = n_samples

    def __getitem__(self, index: int) -> tuple:
        """
        Load the image and ring parameters of the event at the given index.
        Apply the transforms to the image if specified, convert the ring
        parameters to a torch.Tensor and return the image and ring parameters.

        Args
        ----
            index: int
                Index of the sample to load.

        Returns
        -------
            x: torch.Tensor
                Image of the event.
            y: torch.Tensor
                Ring parameters of the event.
        """
        X_dir = self.target_dir / 'X'
        y_dir = self.target_dir / 'y'
        x = cv2.imread(str(X_dir / f'{index}.png'), cv2.IMREAD_GRAYSCALE)
        y = np.load(y_dir / f'{index}.npy').astype(np.float32)

        if self.transforms:
            x = self.transforms(x)
        y = torch.from_numpy(y)

        return x, y

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return self.n_samples


class EventDataModule(pl.LightningDataModule):
    """
    DataModule that loads the training, validation and test datasets.

    Args
    ----
        batch_size: int
            Batch size to use for the dataloaders.
        dataset_sizes: tuple
            Tuple containing the number of samples to load from the training,
            validation and test datasets. If None, all samples are loaded.
    """

    def __init__(self, batch_size: int, dataset_sizes: Optional[tuple] = None):
        super(EventDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.save_hyperparameters()

    def setup(self, stage: str):
        """
        Load the training, validation and test datasets. For testing there are
        two datasets, one for toymodel data and one for simulated data.
        """
        trainsize, valsize, testsize = self.dataset_sizes if self.dataset_sizes else (
            None, None, None)

        self.trainset = EventDataset(DATA_DIR / 'train', n_samples=trainsize,
                                     transforms=self.transforms)
        self.valset = EventDataset(DATA_DIR / 'val', n_samples=valsize,
                                   transforms=self.transforms)

        # define two test sets, one for toymodel data and one for simulated data
        self.dataset_names = ['test', 'sim_data']
        self.testset_1 = EventDataset(DATA_DIR / self.dataset_names[0],
                                      n_samples=testsize,
                                      transforms=self.transforms)
        self.testset_2 = EventDataset(DATA_DIR / self.dataset_names[1],
                                      n_samples=None,
                                      transforms=self.transforms)

    def train_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for the training dataset.
        """
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        """
        Return a DataLoader for the validation dataset.
        """
        return DataLoader(self.valset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

    def test_dataloader(self) -> list[DataLoader]:
        """
        Return a list of DataLoaders for the test datasets.
        """
        loader_1 = DataLoader(self.testset_1, batch_size=self.batch_size,
                              shuffle=False, num_workers=4)
        loader_2 = DataLoader(self.testset_2, batch_size=self.batch_size,
                              shuffle=False, num_workers=4)
        return [loader_1, loader_2]


class ResNet18(torch.nn.Module):
    """
    Implementation of a ResNet18 model with a single channel input. The
    last fully connected layer is replaced by a linear layer with 25
    outputs. The output is reshaped to a 5x5 matrix.
    """

    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)
        # change the input channel to 1
        self.resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = torch.nn.Linear(512, 25)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 5, 5)
        return x


class LitResNet18(pl.LightningModule):
    """
    LightningModule that defines the training, testing and validation behavior.

    Args
    ----
        fit_gt: bool
            If True, the logged samples images are fitted both with the
            predicted as well as the ground truth ring parameters.
    """

    def __init__(self, fit_gt=False):
        super(LitResNet18, self).__init__()
        # define hyperparameters
        # learning rate is only set for tracking purposes
        # it will be overwritten by the scheduler
        self.learning_rate = 0.1
        self.fit_gt = fit_gt
        self.dataset_names = ['test', 'sim_data']

        # define model and loss
        self.model = ResNet18()
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        self.max_lr = self.learning_rate * 25
        self.save_hyperparameters({
            'initial_lr': self.learning_rate,
            'max_lr': self.max_lr
        })

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch. 
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx, dataloader_idx):
        """
        Test step for a single batch. The dataloader_idx is used to
        determine which test dataset is used.

        Here, not only the loss is logged, but also the model inference
        on a few samples of the test dataset. The samples are logged
        as images with ring fits.

        Furthermore, the 5 worst predictions per batch are returned so
        that they can be further processed in the test_epoch_end method.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        # log samples of the model inference
        n_per_batch = 10  # number of samples to log per batch
        if self.fit_gt:
            samples = [plot_single_event(img, Y1=pars, Y2=ideal) for img, pars, ideal
                       in zip(x[:n_per_batch], y_hat[:n_per_batch], y[:n_per_batch])]
        else:
            samples = [plot_single_event(img, Y1=pars) for img, pars
                       in zip(x[:n_per_batch], y_hat[:n_per_batch])]

        self.logger.log_image(  # type: ignore
            key=f'Model inference on dataset: {self.dataset_names[dataloader_idx]}', images=samples)

        # log the 5 worst predictions per batch
        n_worst_per_batch = 5
        losses = [self.loss(y_hat[i], y[i]) for i in range(len(y))]
        losses, indices = torch.sort(torch.stack(losses), descending=True)

        worst = (x[indices[:n_worst_per_batch]],
                 y[indices[:n_worst_per_batch]],
                 y_hat[indices[:n_worst_per_batch]])

        self.log('test_loss', loss)
        return {'test_loss': loss, 'x': worst[0], 'y': worst[1], 'y_hat': worst[2]}

    def test_epoch_end(self, outputs):
        """
        Here, outputs contains the 5 worst predictions of each batch which were
        accumulated in the test_step method. These will be used to log the
        overall 50 worst predictions of the test dataset as images with ring
        fits.
        """
        for idx, name in enumerate(self.dataset_names):
            output = outputs[idx]
            x = torch.cat([out['x'] for out in output])  # type: ignore
            y = torch.cat([out['y'] for out in output])  # type: ignore
            y_hat = torch.cat([out['y_hat'] for out in output])  # type: ignore

            losses = [self.loss(y_hat[i], y[i]) for i in range(len(y))]
            losses, indices = torch.sort(torch.stack(losses), descending=True)

            # log the worst 50 predictions of the whole test set
            n_items = max(50, len(x))
            if self.fit_gt:
                worst_samples = [plot_single_event(img, pars, ideal)
                                 for img, pars, ideal in zip(x[:n_items], y_hat[:n_items], y[:n_items])]
            else:
                worst_samples = [plot_single_event(img, pars)
                                 for img, pars in zip(x[:n_items], y_hat[:n_items])]
            self.logger.log_image(  # type: ignore
                key=f'Worst predictions on dataset: {name}', images=worst_samples)

            # create dataframes for worst predictions
            df1 = pd.DataFrame(y.cpu().numpy().reshape(-1, 25))
            df2 = pd.DataFrame(y_hat.cpu().numpy().reshape(-1, 25))

            fig = ring_params_hist(df1.to_numpy(), silent=True)
            fig = ring_params_hist(df2.to_numpy(), silent=True)

            wandb.log({f'Ring parameters on dataset: {name}': fig})
            wandb.log({f'Predicted ring parameters on dataset: {name}': fig})
            # log dataframes
            wandb.log(
                {f'Ring parameters on dataset: {name}': wandb.Table(dataframe=df1)})
            wandb.log(
                {f'Predicted ring parameters on dataset: {name}': wandb.Table(dataframe=df2)})

    def configure_optimizers(self):
        """
        Define a SGD optimizer with a OneCycleLR scheduler and return it.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            nesterov=True,
            lr=self.learning_rate,
            momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer,
            max_lr=self.max_lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


if __name__ == '__main__':
    # define hyperparameters
    batch_size = 2000
    n_epochs = 20
    # dataset_sizes = (1000, None, None)
    dataset_sizes = (None, None, None)

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--version', type=int, default=None)
    args = parser.parse_args()
    evaluate = args.eval
    version = args.version

    if evaluate and version is not None:
        try:
            ckpt_path = list(
                (MODEL_DIR / f'version_{version}' / 'checkpoints').glob('*.ckpt'))[0]
            print(f'Using checkpoint {ckpt_path} for evaluation...')
            VERSION = version
        except IndexError:
            print(
                f'ValueError: No checkpoint found in {MODEL_DIR} for version {version}. Exiting...')
            sys.exit(1)
    elif evaluate and version is None:
        ckpt_path = list(
            (MODEL_DIR / 'latest' / 'checkpoints').glob('*.ckpt'))[0]
        print(f'Using checkpoint {ckpt_path} for evaluation...')

    # define model and datamodule
    model = LitResNet18() if not evaluate else LitResNet18.load_from_checkpoint(ckpt_path,  # type: ignore
                                                                                batch_size=batch_size,
                                                                                dataset_sizes=dataset_sizes,
                                                                                fit_gt=True)
    dm = EventDataModule(batch_size=batch_size, dataset_sizes=dataset_sizes)

    # define logger, callbacks and trainer
    wandb_logger = WandbLogger(
        project='models',
        save_dir=ROOT_DIR,
        id=f'{NAME}{VERSION}',
        name=f'{NAME}{VERSION}',
        log_model=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(accelerator='gpu',
                      devices=1,
                      max_epochs=n_epochs,
                      auto_lr_find=not evaluate,
                      #   auto_scale_batch_size='binsearch',
                      callbacks=[lr_monitor],
                      logger=wandb_logger)

    # train model
    if not evaluate:
        trainer.tune(model, datamodule=dm)
        trainer.fit(model, datamodule=dm)
        model.to_onnx(MODEL_DIR / f'{NAME}{VERSION}' / 'model.onnx',
                      input_sample=torch.randn(10, 1, 72, 32))
        # create symbolic link to latest model
        source_path = MODEL_DIR / f'version_{VERSION}'
        link_path = MODEL_DIR / 'latest'
        if link_path.exists():
            link_path.unlink()
        link_path.symlink_to(source_path)

    # evaluate model
    print('Evaluating model...')
    t = time.time()
    trainer.test(model, datamodule=dm)
    t = time.time() - t
    print(f'Testing took {t:.2f} seconds to run.')
