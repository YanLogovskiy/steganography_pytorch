import torch
import random
import numpy as np

from itertools import product
from typing import Sequence, List


def generate_random_key(shape, length):
    pool = list(product(*[range(x) for x in shape]))
    return random.sample(pool, length)


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


def calculate_sine_rung(x: torch.tensor, bit_value: int, beta=15):
    # bs, c, h, w
    assert len(x.shape) == 4
    assert bit_value in [0, 1]
    return torch.sigmoid(beta * torch.sin(np.pi * (bit_value - x)))


def calculate_multiplier(x: torch.tensor, eps: float = 2e-7):
    plus_eps_mask = (x >= -1) & (x < -1 + eps)
    minus_eps_mask = (x <= 1) & (x > 1 + eps)

    # fill -1, 1
    total_mask = np.random.choice([-1, 1], np.prod(x.shape)).reshape(x.shape)
    total_mask = torch.from_numpy(total_mask)
    total_mask[plus_eps_mask] = 1
    total_mask[minus_eps_mask] = -1
    values = total_mask * eps
    return values

# # TODO: optimize
# def S_m(m, x):
#     if m == 1 and x >= -1 and x < -1 + (2 ** -7):
#         ksi = (2 ** -7)
#     elif m == 0 and x > 1 - (2 ** -7) and x <= 1:
#         ksi = -(2 ** -7)
#     else:
#         ksi = random.choice([-(2 ** -7), (2 ** -7)])
#
#     z = (x + 1) * (2 ** 7)
#
#     return x + ksi * p_m(m, z)
#
#
# def LSB(image, message):
#     img = image
#
#     message_size = len(message)
#     axis_0 = random.sample(range(image.shape[1]), message_size)
#     axis_1 = random.sample(range(image.shape[2]), message_size)
#
#     for pixel_i, pixel_j, i in zip(axis_0, axis_1, range(len(message))):
#         img[2][pixel_i][pixel_j] = S_m(message[i], img[2][pixel_i][pixel_j])
#
#     return img
