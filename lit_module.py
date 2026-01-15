import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy
from model import ResNet18CIFAR10
from config import Config


class CIFAR10LitModule(pl.LightningModule):
    def __init__(self, learning_rate: float = Config.LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ResNet18CIFAR10(num_classes=Config.NUM_CLASSES)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task='multiclass', num_classes=Config.NUM_CLASSES)
        self.val_acc = Accuracy(task='multiclass', num_classes=Config.NUM_CLASSES)
        self.test_acc = Accuracy(task='multiclass', num_classes=Config.NUM_CLASSES)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
