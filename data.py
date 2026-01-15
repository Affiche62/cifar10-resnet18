import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from config import Config


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = Config.DATA_DIR, batch_size: int = Config.BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=Config.RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip(p=Config.RANDOM_FLIP_PROB),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.CIFAR10_MEAN, std=Config.CIFAR10_STD)
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.CIFAR10_MEAN, std=Config.CIFAR10_STD)
        ])
    
    def setup(self, stage: str | None = None):
        if stage == 'fit' or stage is None:
            self.cifar10_train = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.train_transform
            )
            self.cifar10_val = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.test_transform
            )
        
        if stage == 'test' or stage is None:
            self.cifar10_test = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.test_transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
