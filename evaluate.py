import time
import os

import torch
from torch import nn

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver, QConfig
from torch.ao.quantization import quantize_fx

from model import *
from main import train_step
from data_utils import get_data, get_dataloaders, accuracy_fn
from quantization_utils import print_size_of_model

from tqdm.auto import tqdm

# Configurations
BATCH_SIZE=32
EPOCHS=10


def load_model(model:torch.nn.Module, PATH: str, device:str) -> nn.Module: 
    """
        Loads a given model from a specified path onto a device
        Returns: 
        - model (nn.Module) : Loaded model 
    """
    device = torch.device(device)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    return model 

def print_size_of_model(model):
    """
        Prints the size of the model in MB.
    """
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    inf_time = 0.0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            start_time= time.time()
            y_pred = model(X)
            end_time = time.time() - start_time
            inf_time += end_time
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc
        loss /= len(data_loader)
        inf_time /= len(data_loader)
        acc /= len(data_loader)
        print_size_of_model(model)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc(%)": acc,
            "model_time(ms)": inf_time*1000}

def main():
    # create dataset & dataloader
    train_data, test_data = get_data()
    train_dataloader, test_dataloader = get_dataloaders(train_data, test_data)

    # load model
    file_path = "./models/resnet5_flp.pth"
    model = ResNet5(1, ResBlock, outputs=10)
    flp_model = load_model(model, file_path, "cpu")

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
    
    # run inference on fp32 model
    flp_results = eval_model(model=flp_model,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device="cpu")
    print(flp_results)
   
    # ptq 
    """
    symmetric_qconfig = torch.ao.quantization.QConfig(
    activation=FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric))

    asymmetric_qconfig = torch.ao.quantization.QConfig(
        activation=FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_affine))
    
    qconfig_mapping = QConfigMapping().set_global(symmetric_qconfig)
    flp_model.eval()
    example_inputs = (next(iter(test_dataloader))[0])
    prepared_model = prepare_fx(flp_model, qconfig_mapping, example_inputs)
    
    with torch.no_grad():
        for image, _ in test_dataloader:
            prepared_model(image)
    
    model_quantized = quantize_fx.convert_fx(prepared_model)
    qmodel_results = eval_model(model=model_quantized, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn, device="cpu")
    print(qmodel_results)
    """

if __name__ == "__main__":
    main()