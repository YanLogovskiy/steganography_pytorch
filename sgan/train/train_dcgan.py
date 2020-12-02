import os
import torch
import argparse
import numpy as np

from pathlib import Path
from torch.optim import Adam
from torch.nn import functional as F
from typing import Sequence, Callable

from sgan.train.loggers import TBLogger
from sgan.modules import Generator, Discriminator
from sgan.data import CelebDataset, split_data, create_iterators
from sgan.train.base import process_batch, generate_noise, inference_step, to_numpy, save_numpy, save_torch


def train_dcgan(*, generator, discriminator, train_iterator, device, n_epoch,
                n_batches_per_epoch, batch_size, generator_opt, discriminator_opt, n_noise_channels,
                callbacks: Sequence[Callable] = None, logger: TBLogger):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion = F.binary_cross_entropy_with_logits

    callbacks = callbacks or []
    for epoch in range(n_epoch):
        generator_losses = []
        discriminator_losses_on_real = []
        discriminator_losses_on_fake = []

        for _ in range(n_batches_per_epoch):
            discriminator_opt.zero_grad()

            # train discriminator on real
            real_batch, _ = next(train_iterator)
            real_target = torch.ones(batch_size, 1, 1, 1)
            real_loss = process_batch(real_batch, real_target, discriminator, criterion)

            # train discriminator on fake
            noise = generate_noise(batch_size, n_noise_channels)
            fake_batch = generator(noise)
            target_for_discriminator = torch.zeros(batch_size, 1, 1, 1)
            fake_loss = process_batch(fake_batch.detach(), target_for_discriminator, discriminator, criterion)
            discriminator_opt.step()

            generator_opt.zero_grad()
            # train generator
            target_for_generator = torch.ones(batch_size, 1, 1, 1)
            generator_loss = process_batch(fake_batch, target_for_generator, discriminator, criterion)
            generator_opt.step()

            generator_losses.append(generator_loss)
            discriminator_losses_on_real.append(real_loss)
            discriminator_losses_on_fake.append(fake_loss)

        # run callbacks
        for callback in callbacks:
            callback(epoch)

        losses = {'Generator': np.mean(generator_losses),
                  'Discriminator on fake': np.mean(discriminator_losses_on_fake),
                  'Discriminator on real': np.mean(discriminator_losses_on_real)
                  }
        logger.policies(losses, epoch)


def run_experiment(*, device, download: bool, train_size: bool, val_size: bool, test_size: bool, n_epoch: int,
                   batch_size: int, n_batches_per_epoch: int, n_noise_channels: int, save_path: str):
    # path to save everything related to experiment
    save_path = Path(save_path).expanduser()

    # create dataset and models
    dataset = CelebDataset(download=download)
    generator = Generator(in_channels=n_noise_channels).to(device)
    discriminator = Discriminator().to(device)

    split = split_data(dataset, train_size=train_size, val_size=val_size, test_size=test_size)
    train_indices, val_indices, test_indices = split
    train_iterator, val_iterator = create_iterators(dataset, train_indices, val_indices, batch_size=batch_size)

    # TODO: remove hardcode
    optimizer_parameters = dict(lr=1e-4, betas=(0.5, 0.99))
    generator_opt = Adam(generator.parameters(), **optimizer_parameters)
    discriminator_opt = Adam(discriminator.parameters(), **optimizer_parameters)

    fixed_noise = torch.randn(64, n_noise_channels, 1, 1, device=device)

    def predict_on_fixed_noise(epoch, prefix='fixed_noise', compression=1):
        predict = to_numpy(inference_step(fixed_noise, generator))
        os.makedirs(save_path / prefix, exist_ok=True)
        save_numpy(predict, save_path / prefix / f'{epoch}.npy.gz', compression=compression)

    def save_models(epoch):
        save_torch(generator, save_path / 'generator')
        save_torch(discriminator, save_path / 'discriminator')

    logger = TBLogger(save_path / 'logs')
    callbacks = [predict_on_fixed_noise, save_models]
    train_dcgan(
        generator=generator,
        generator_opt=generator_opt,
        discriminator=discriminator,
        discriminator_opt=discriminator_opt,
        train_iterator=train_iterator,
        device=device,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_batches_per_epoch=n_batches_per_epoch,
        n_noise_channels=n_noise_channels,
        callbacks=callbacks,
        logger=logger
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')

    parser.add_argument('--train_size', default=0.6, type=float)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--val_size', default=0.2, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n_epoch', default=2, type=int)
    parser.add_argument('--n_batches_per_epoch', default=5, type=int)
    parser.add_argument('--n_noise_channels', default=64, type=int)
    parser.add_argument('--save_path', default='~/celeba', type=str)

    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == '__main__':
    main()
