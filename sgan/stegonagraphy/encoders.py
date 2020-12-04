import numpy as np
from sgan.stegonagraphy.utils import bits_to_bytes


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


class SigmoidPytorchEncoder(LeastSignificantBitEncoder):
    def encode(self, container: np.ndarray, message: np.ndarray, key: np.ndarray):
        pass

    def decode(self, container: np.ndarray, key: np.ndarray):
        pass

    def encode_batch(self):
        raise NotImplementedError
