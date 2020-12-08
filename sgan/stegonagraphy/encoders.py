import torch
import numpy as np
from sgan.stegonagraphy.utils import bits_to_bytes, calculate_sine_step, calculate_multiplier


class LeastSignificantBitEncoder:
    def __init__(self, convert_to_bytes=False):
        self.convert_to_bytes = convert_to_bytes

    # methods used for a single image
    def encode(self, container, message, key):
        raise NotImplementedError

    def decode(self, container, key):
        raise NotImplementedError

    def encode_batch(self, containers, messages, keys):
        raise NotImplementedError

    def decode_batch(self, containers, keys):
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

    # TODO: vectorize
    def encode_batch(self, containers: np.ndarray, messages: np.ndarray, keys: np.ndarray):
        result = []
        for container, message, key in zip(containers, messages, keys):
            result.append(self.encode(container, message, key))
        return np.stack(result)

    def decode_batch(self, containers: np.ndarray, keys: np.ndarray):
        result = []
        for container, key in zip(containers, keys):
            result.append(self.decode(container, key))
        return result


class SigmoidTorchEncoder(LeastSignificantBitEncoder):
    def __init__(self, *args, beta=5, inv_eps=128, **kwargs):
        self.inv_eps = inv_eps
        self.beta = beta
        super().__init__(*args, **kwargs)

    def encode(self, container: torch.tensor, message: np.ndarray, key: np.ndarray):
        assert len(container.shape) == 3
        assert len(message) == len(key)
        # some calculations
        container = torch.clone(container)
        red_channel = container[0, ...]
        scaled_channel = self.inv_eps * (red_channel + 1)
        # calculate sine approximation
        one_step = calculate_sine_step(scaled_channel, 1, beta=self.beta)
        zero_step = calculate_sine_step(scaled_channel, 0, beta=self.beta)
        # random multipliers
        one_mul = calculate_multiplier(red_channel, 1, inv_eps=self.inv_eps)
        zero_mul = calculate_multiplier(red_channel, 0, inv_eps=self.inv_eps)
        one_addition = one_mul * one_step
        zero_addition = zero_mul * zero_step
        # indices for message embedding
        one_pos = torch.tensor([x for i, x in enumerate(key) if message[i] == 1])
        zero_pos = torch.tensor([x for i, x in enumerate(key) if message[i] == 0])
        # embed message
        red_channel[one_pos[:, 0], one_pos[:, 1]] += one_addition[one_pos[:, 0], one_pos[:, 1]]
        red_channel[zero_pos[:, 0], zero_pos[:, 1]] += zero_addition[zero_pos[:, 0], zero_pos[:, 1]]
        return torch.clamp(container, -1, 1)

    def decode(self, container: torch.tensor, key: torch.tensor):
        assert len(container.shape) == 3
        red_channel = self.inv_eps * (container[0, ...] + 1)
        red_channel = np.floor(red_channel.numpy()).astype(np.int)

        message = []
        for pos in key:
            message.append(red_channel[pos] & 1)

        if self.convert_to_bytes:
            message = bits_to_bytes(message)
        return message

    def encode_batch(self, containers: torch.tensor, messages: np.ndarray, keys: np.ndarray):
        result = []
        for container, message, key in zip(containers, messages, keys):
            result.append(self.encode(container, message, key))
        return torch.stack(result)

    def decode_batch(self, containers: torch.tensor, keys: np.ndarray):
        result = []
        for container, key in zip(containers, keys):
            result.append(self.decode(container, key))
        return result
