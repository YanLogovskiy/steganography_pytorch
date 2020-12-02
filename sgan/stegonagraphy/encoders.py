import numpy as np


def generate_random_pairs(low, high, length, seed=None):
    if seed is None:
        pass
    else:
        pass


def bits_to_bytes():
    pass


def bytes_to_bits():
    pass


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
