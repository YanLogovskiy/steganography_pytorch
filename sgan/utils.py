import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from pathlib import Path

from typing import Callable, Union


def get_device(x):
    if isinstance(x, nn.Module):
        try:
            return next(x.parameters()).device
        except StopIteration:
            raise ValueError('No parameters inside module')
    elif isinstance(x, torch.Tensor):
        return x.device
    else:
        raise RuntimeError('Wrong input')


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def process_batch(inputs: torch.Tensor, target: torch.Tensor, model: nn.Module,
                  criterion: Callable):
    model.train()
    inputs = inputs.to(device=get_device(model))
    target = target.to(device=get_device(model))

    predict = model(inputs)
    loss = criterion(predict, target)
    loss.backward()
    return to_numpy(loss)


def generate_noise(batch_size, n_channels, device):
    return torch.randn(batch_size, n_channels, 1, 1, device=device)


def inference_step(inputs, model):
    model.eval()
    inputs = inputs.to(get_device(model))
    with torch.no_grad():
        return model(inputs)


def scale_gradients(model: nn.Module, scale):
    for p in model.parameters():
        p.grad *= scale


PathLike = Union[Path, str]


def save_torch(o: nn.Module, path: PathLike):
    torch.save(o.state_dict(), path)


mean = torch.tensor((0.5, 0.5, 0.5))
std = torch.tensor((0.5, 0.5, 0.5))
mean = mean.reshape(1, 3, 1, 1)
std = std.reshape(1, 3, 1, 1)


def transform_gan(x: torch.tensor):
    m = mean.to(get_device(x))
    s = std.to(get_device(x))
    x = x / 255
    return (x - m) / s


def inverse_transform_gan(x: torch.tensor):
    m = mean.to(get_device(x))
    s = std.to(get_device(x))
    x = x * s + m
    x = x * 255
    return x


def transform_encoder(x: torch.tensor):
    return x / 127.5 - 1


def inverse_transform_encoder(x: torch.tensor):
    return 127.5 * (x + 1)
