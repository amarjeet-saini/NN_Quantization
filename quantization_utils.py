import os
import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver, QConfig
from torch.ao.quantization import quantize_fx

# Define quantization configuration
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

qconfig = ("symmetric", "asymmetric")

def quantize(model:nn.Module, test_dataloader, qconfig:str):
    """
    Quantizes a given model using specified quantization configuration.
    
    Args:
        model (nn.Module): PyTorch model to be quantized.
        test_dataloader (DataLoader): DataLoader for calibration.
        config (str): Configuration name ("symmetric" or "asymmetric").
        
    Returns:
        Quantized model (nn.Module): Quantized model. 
    """
    if qconfig == "symmetric":
        qconfig_mapping = QConfigMapping().set_global(symmetric_qconfig)
    elif qconfig == "asymmetric":
        qconfig_mapping = QConfigMapping().set_global(asymmetric_qconfig)
    else:
        raise ValueError("Unsupported config: {qconfig}")

    model.eval()
    example_input = (next(iter(test_dataloader))[0])
    
    # convert to torch.fx
    prepared_model = prepare_fx(model, qconfig_mapping, example_input)

    # calibrate the model - learn qparams
    with torch.no_grad():
        for image, _ in test_dataloader:
            prepared_model(image)
    
    # quantize model (operation converted to int8)
    model_quantized = quantize_fx.convert_fx(prepared_model)
    return model_quantized

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