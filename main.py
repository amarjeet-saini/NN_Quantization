from model import *

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Hyper-parameters
BATCH_SIZE=32
EPOCHS=3

def get_data():
    train_data = datasets.MNIST(root="data",
                                transform=ToTensor(), download=True)
    test_data = datasets.MNIST(root="data", train=False,
                               transform=ToTensor(), download=True)
    print(f"Number of train images: {len(train_data)}, test images: {len(test_data)}") 
    print(f"Channel_Height_Width: {train_data[0][0].shape}")
    return train_data, test_data

def get_dataloaders(train_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader

def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X) 
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    
    train_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Train accuracy: {(100*correct):>0.1f}% | Train loss: {train_loss:.3f}")

def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module):
    test_loss = 0
    model.eval()
    with torch.inference(): 
        for X,y in dataloader:
            y_pred = model(X) 
            loss = loss_fn(y_pred, y)
            test_loss += loss
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
        
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
    print(f"Test accuracy: {(100*correct):>0.1f}% | Train loss: {test_loss:.3f}")
 
def main():
    
    # Step1 : Data
    train_data, test_data = get_data()

    # Step2 : Prepare Dataloader
    train_dataloader, test_dataloader = get_dataloaders(train_data, test_data)

    # Step3: Build model
    model = ResNet5(1, ResBlock, outputs=10)

    # Step4: Train model
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

    # Step 5: Evaluate model
    
    for epoch in tqdm(range(1,EPOCHS+1)):
        print(f"Epoch: {epoch}\n---------")
        train_step(model, train_dataloader, loss_fn, optim) 
        test_step(model, test_dataloader, loss_fn)
    
    # Step6: Save model
    #export(model, "resnet5_fxp" + str(datetime.now().second) + ".onnx")

if __name__ == "__main__":
    main()
    