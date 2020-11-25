import torch

from torch.nn import functional as F
from torch.optim import Adam
from sgan.data import CelebDataset, split_data, create_iterators
from sgan.modules import Generator, Discriminator

from sgan.train.base import process_batch


def generate_noise(batch_size, n_channels):
    return torch.randn(batch_size, n_channels, 1, 1)


def inference_step(inputs, model):
    with torch.no_grad():
        return model(inputs)


def train_dcgan(*, download=False, device='cuda', n_epoch=100, n_batches_per_epoch=100, lr=1e-4, beta1=0.5, beta2=0.99,
                train_size=0.6, val_size=0.2, test_size=0.2, noise_n_channels=64, batch_size=128):
    dataset = CelebDataset(download=download)
    train_indices, val_indices, test_indices = split_data(dataset, train_size=train_size, val_size=val_size,
                                                          test_size=test_size)
    train_iterator, val_iterator = create_iterators(dataset, train_indices, val_indices)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = F.binary_cross_entropy_with_logits
    optimizer_parameters = dict(lr=lr, betas=(beta1, beta2))
    generator_opt = Adam(generator.parameters(), **optimizer_parameters)
    discriminator_opt = Adam(discriminator.parameters(), **optimizer_parameters)

    for epoch in range(n_epoch):

        for _ in range(n_batches_per_epoch):
            discriminator_opt.zero_grad()

            # train discriminator on real
            real_batch = next(train_iterator)
            real_target = torch.ones(batch_size)
            real_loss = process_batch(real_batch, real_target, discriminator, criterion)

            # train discriminator on fake
            noise = generate_noise(batch_size, noise_n_channels)
            fake_batch = generator(noise)
            fake_target = torch.zeros(batch_size)
            noise_loss = process_batch(fake_batch, fake_target, discriminator, criterion)

            discriminator_opt.step()

            generator_opt.zero_grad()
            # train generator
            target = torch.ones(batch_size)
            generator_loss = process_batch(fake_batch, target, discriminator, criterion)
            generator_opt.step()


def main():
    pass
