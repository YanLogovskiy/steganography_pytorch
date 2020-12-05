import torch
import functools
import numpy as np

from torch import nn
from math import ceil
from pathlib import Path
from gzip import GzipFile

from typing import Callable, Union
from joblib import Parallel, delayed


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
    with torch.no_grad():
        return model(inputs)


def scale_gradients(model: nn.Module, scale):
    for p in model.parameters():
        p.grad *= scale


PathLike = Union[Path, str]


def save_torch(o: nn.Module, path: PathLike):
    torch.save(o.state_dict(), path)


def save_numpy(value, path: PathLike, *, allow_pickle: bool = True, fix_imports: bool = True, compression: int = None):
    if compression is not None:
        with GzipFile(path, 'wb', compresslevel=compression) as file:
            return save_numpy(value, file, allow_pickle=allow_pickle, fix_imports=fix_imports)

    np.save(path, value, allow_pickle=allow_pickle, fix_imports=fix_imports)


def load_numpy(path: PathLike, *, allow_pickle: bool = True, fix_imports: bool = True, decompress: bool = False):
    if decompress:
        with GzipFile(path, 'rb') as file:
            return load_numpy(file, allow_pickle=allow_pickle, fix_imports=fix_imports)

    return np.load(path, allow_pickle=allow_pickle, fix_imports=fix_imports)