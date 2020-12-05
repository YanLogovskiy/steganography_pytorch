import torch
import random
import numpy as np

from itertools import product
from functools import lru_cache
from typing import Sequence, List

from sgan.utils import to_numpy


def generate_random_key(shape, length):
    pool = list(product(*[range(x) for x in shape]))
    return random.sample(pool, length)


@lru_cache(None)
def bytes_to_bits(sequence: Sequence):
    bits = []
    # looks ugly :(
    for ch in sequence:
        str_bits = bin(ord(ch)).lstrip('0b')
        while len(str_bits) != 8:
            str_bits = '0' + str_bits
        bits.extend(list(map(int, str_bits)))

    return bits


def bits_to_bytes(sequence: List[int]):
    chunks = [sequence[i:i + 8] for i in range(0, len(sequence), 8)]
    # pad if necessary
    while len(chunks[-1]) != 8:
        chunks[-1].append(0)

    def process_single(chunk):
        result = 0
        for i, x in enumerate(reversed(chunk)):
            result += x * 2 ** i
        return chr(result)

    message = [process_single(ch) for ch in chunks]
    return message


def calculate_sine_step(x: torch.tensor, bit_value: int, beta=15):
    assert bit_value in [0, 1]
    return torch.sigmoid(beta * torch.sin(np.pi * (bit_value - x)))


def calculate_multiplier(x: torch.tensor, bit_value, inv_eps: float = 128):
    eps = 1 / inv_eps
    total_mask = np.random.choice([-1, 1], np.prod(x.shape)).reshape(x.shape)
    if bit_value == 1:
        plus_eps_mask = (x >= -1) & (x < -1 + eps)
        plus_eps_mask = to_numpy(plus_eps_mask)
        total_mask[plus_eps_mask] = 1
    elif bit_value == 0:
        minus_eps_mask = (x > 1 - eps) & (x <= 1)
        minus_eps_mask = to_numpy(minus_eps_mask)
        total_mask[minus_eps_mask] = -1

    total_mask = torch.from_numpy(total_mask).to(x)
    values = total_mask * eps
    return values
