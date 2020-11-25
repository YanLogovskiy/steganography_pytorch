import numpy as np
import random

# TODO: move to pytorch
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def p_m(m, z):
    beta = 1  # fidelity of the approximation - why so big (2e15) ?

    return sigmoid(beta * np.sin((z - m) * np.pi))


# TODO: optimize
def S_m(m, x):
    if m == 1 and x >= -1 and x < -1 + (2 ** -7):
        ksi = (2 ** -7)
    elif m == 0 and x > 1 - (2 ** -7) and x <= 1:
        ksi = -(2 ** -7)
    else:
        ksi = random.choice([-(2 ** -7), (2 ** -7)])

    z = (x + 1) * (2 ** 7)

    return x + ksi * p_m(m, z)


def LSB(image, message):
    img = image

    message_size = len(message)
    axis_0 = random.sample(range(image.shape[1]), message_size)
    axis_1 = random.sample(range(image.shape[2]), message_size)

    for pixel_i, pixel_j, i in zip(axis_0, axis_1, range(len(message))):
        img[2][pixel_i][pixel_j] = S_m(message[i], img[2][pixel_i][pixel_j])

    return img
