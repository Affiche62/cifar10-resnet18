import torch
import torch.nn as nn
from torchvision.models import resnet18
from config import Config


def create_resnet18_for_cifar10(num_classes: int = Config.NUM_CLASSES):
    model = resnet18(weights=None)
    
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    model.maxpool = nn.Identity()
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes: int = Config.NUM_CLASSES):
        super().__init__()
        self.model = create_resnet18_for_cifar10(num_classes)
    
    def forward(self, x):
        return self.model(x)
