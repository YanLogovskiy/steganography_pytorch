import torch
from torch import nn

from typing import Callable, Sequence


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

    loss = criterion(model(inputs), target)
    loss.backward()
    return to_numpy(loss)
