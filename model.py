import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime 


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels) 

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = F.relu(self.bn1(self.conv1(input))) 
        input = F.relu(self.bn2(self.conv2(input))) 
        input = input + shortcut
        input = F.relu(input)
        return input 
    
class ResNet5(nn.Module):
    """
    Resnet-5 (Hyperparameters from Resnet-9 architecture)
    Reference: https://github.com/matthias-wright/cifar10-resnet 
    """
    def __init__(self, in_channels, resblock, outputs=10):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential( 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) 
        
        self.layer2 = nn.Sequential(
            resblock(128,128)
        )
        
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(128, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)
        return input

def export(model:nn.Module, filename:str):
    dummy_input = (torch.randn(1, 1, 28, 28))
    torch.onnx.export(model, dummy_input, filename)

"""
def main():
    model = ResNet5(1, ResBlock, outputs=10)
    export(model, "resnet5_fxp" + str(datetime.now().second) + ".onnx")

if __name__ == "__main__":
    main()
"""