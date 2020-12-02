import numpy as np
import random
from itertools import product
from typing import Sequence, List


def generate_random_key(image, length):
    h, w = image.shape[1:]
    pool = list(product(range(h), range(w)))
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

    print(chunks)

    def process_single(chunk):
        result = 0
        for i, x in enumerate(reversed(chunk)):
            result += x * 2 ** i
        return chr(result)

    message = [process_single(ch) for ch in chunks]
    return message


class LeastSignificantBitEncoder:
    def __init__(self, convert_to_bytes=False):
        self.convert_to_bytes = convert_to_bytes

    # methods used for a single image
    def encode(self, container, message, key):
        raise NotImplementedError

    def decode(self, container, key):
        raise NotImplementedError


class PlusMinusNumpyEncoder(LeastSignificantBitEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, container: np.ndarray, message: np.ndarray, key: np.ndarray):
        assert len(message) == len(key)
        assert len(container.shape) == 3

        encoded = container.copy()
        # extract red channel
        red_channel = encoded[0, ...]
        image_shape = red_channel.shape

        lsb_random_map = np.reshape(np.random.choice([-1, 1], np.prod(image_shape)), image_shape)
        for bit_value, pos in zip(message, key):
            if bit_value != red_channel[pos] & 1:
                red_channel[pos] += lsb_random_map[pos]

        return encoded

    def decode(self, container: np.ndarray, key: np.ndarray):
        assert len(container.shape) == 3
        red_channel = container[0, ...]

        message = []
        for pos in key:
            message.append(red_channel[pos] & 1)

        if self.convert_to_bytes:
            message = bits_to_bytes(message)
        return message


class SigmoidNumpyEncoder(LeastSignificantBitEncoder):
    def encode(self, container: np.ndarray, message: np.ndarray, key: np.ndarray):
        pass

    def decode(self, container: np.ndarray, key: np.ndarray):
        pass


class PlusMinusPytorchEncoder(LeastSignificantBitEncoder):
    def encode(self, container: np.ndarray, message: np.ndarray, key: np.ndarray):
        pass

    def decode(self, container: np.ndarray, key: np.ndarray):
        pass


class SigmoidPytorchEncoder(LeastSignificantBitEncoder):
    def encode(self, container: np.ndarray, message: np.ndarray, key: np.ndarray):
        pass

    def decode(self, container: np.ndarray, key: np.ndarray):
        pass

# # TODO: move to pytorch
# def sigmoid(X):
#     return 1 / (1 + np.exp(-X))
#
#
# def p_m(m, z):
#     beta = 1  # fidelity of the approximation - why so big (2e15) ?
#
#     return sigmoid(beta * np.sin((z - m) * np.pi))
#
#
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
