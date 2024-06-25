import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 32

def get_data():
    """
    Download and returns MNIST dataset

    Returns:
        training_data (Dataset), testing_data (Dataset)
    """
    train_data = datasets.MNIST(root="data",
                                transform=ToTensor(), download=True)
    test_data = datasets.MNIST(root="data", train=False,
                               transform=ToTensor(), download=True)
    print(f"Number of train images: {len(train_data)}, test images: {len(test_data)}")
    print(f"Channel_Height_Width: {train_data[0][0].shape}")
    return train_data, test_data

def get_dataloaders(train_data, test_data, batch_size=BATCH_SIZE):
    """
    Creates dataloaders(i.e. iterables) for the given datasets 
    Returns:
        train_dataloader, test_dataloader
    """
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader

def accuracy_fn(y_true, y_pred):
    """
    Calculates the accuracy of predictions
    
    Parameters:
    - y_true (torch.Tensor): True labels
    - y_pred (torch.Tensor): Predicted labels
    
    Returns:
    - acc (float): Accuracy in percentage
    """
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100
    return acc