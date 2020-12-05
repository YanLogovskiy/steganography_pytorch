import os
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from torch.nn import functional as F
from typing import Sequence, Callable

from sgan.train.loggers import TBLogger
from sgan.modules import Generator, Discriminator
from sgan.stegonagraphy import SigmoidTorchEncoder, generate_random_key, bytes_to_bits
from sgan.data import CelebDataset, TextIterator, split_data, create_iterators
from sgan.train.utils import process_batch, generate_noise, inference_step, to_numpy, save_numpy, save_torch, \
    scale_gradients


# TODO: too much code repetitions
def run_experiment(*, device, download: bool, train_size: bool, val_size: bool, test_size: bool,
                   n_epoch: int, batch_size: int, n_noise_channels: int, data_path: str, experiment_path: str,
                   embedding_fidelity: float, alpha_balance):
    # path to save everything related to experiment
    data_path = Path(data_path).expanduser()
    experiment_path = Path(experiment_path).expanduser()

    # create dataset and models
    dataset = CelebDataset(root=data_path, download=download)
    discriminator = Discriminator().to(device)
    stego_analyzer = Discriminator().to(device)
    generator = Generator(in_channels=n_noise_channels).to(device)

    split = split_data(dataset, train_size=train_size, val_size=val_size, test_size=test_size)
    train_indices, val_indices, test_indices = split
    train_iterator, val_iterator = create_iterators(dataset, train_indices, val_indices, batch_size=batch_size)

    optimizer_parameters = dict(lr=2e-4, betas=(0.5, 0.99))
    generator_opt = Adam(generator.parameters(), **optimizer_parameters)
    image_analyser_opt = Adam(discriminator.parameters(), **optimizer_parameters)
    message_analyser_opt = Adam(stego_analyzer.parameters(), **optimizer_parameters)

    fixed_noise = torch.randn(64, n_noise_channels, 1, 1, device=device)

    def predict_on_fixed_noise(epoch, prefix='fixed_noise', compression=1):
        predict = to_numpy(inference_step(fixed_noise, generator))
        os.makedirs(experiment_path / prefix, exist_ok=True)
        save_numpy(predict, experiment_path / prefix / f'{epoch}.npy.gz', compression=compression)

    def save_models(epoch):
        save_torch(generator, experiment_path / 'generator')
        save_torch(discriminator, experiment_path / 'discriminator')
        save_torch(stego_analyzer, experiment_path / 'stego_analyser')

    logger = TBLogger(experiment_path / 'logs')
    epoch_callbacks = [predict_on_fixed_noise, save_models]


def train_sgan(*, generator, discriminator, stego_analyser, encoder, text_iterator, image_iterator, device,
               n_epoch, generator_opt, discriminator_opt, stego_opt, n_noise_channels, alpha_balance, callbacks,
               logger):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    stego_analyser = stego_analyser.to(device)
    criterion = F.binary_cross_entropy_with_logits

    callbacks = callbacks or []
    for epoch in tqdm(range(n_epoch)):
        generator_losses = []

        discriminator_losses_on_real = []
        discriminator_losses_on_fake = []

        stego_loss = []

        with image_iterator as iterator:
            for real_batch, _ in iterator:
                batch_size = len(real_batch)
                discriminator_opt.zero_grad()
                # train discriminator on real
                real_target = torch.ones(batch_size, 1, 1, 1)
                real_loss = process_batch(real_batch, real_target, discriminator, criterion)
                # train discriminator on fake
                fake_batch = generator(generate_noise(batch_size, n_noise_channels, device))
                target_for_discriminator = torch.zeros(batch_size, 1, 1, 1)
                fake_loss = process_batch(fake_batch.detach(), target_for_discriminator, discriminator, criterion)
                discriminator_opt.step()
                # train generator
                generator_opt.zero_grad()
                target_for_generator = torch.ones(batch_size, 1, 1, 1)
                generator_loss = process_batch(fake_batch, target_for_generator, discriminator, criterion)
                # rescale gradients
                scale_gradients(generator, alpha_balance)
                generator_opt.step()

                # TODO: start after several epochs?
                # start second part
                containers = generator(generate_noise(batch_size, n_noise_channels, device))
                labels = np.random.choice([0, 1], (batch_size, 1, 1, 1))

                encoded_batch = []
                for container, label in zip(containers, labels):
                    if label == 1:
                        msg = bytes_to_bits(next(text_iterator))
                        key = generate_random_key(container.shape[1:], len(msg))
                        container = encoder.encode(container, msg, key)
                    encoded_batch.append(container)

                encoded_batch = torch.stack(encoded_batch)
                labels = torch.from_numpy(labels).float()
                # train analyser
                stego_opt.zero_grad()
                stego_loss = process_batch(encoded_batch, labels, stego_analyser, criterion)
                stego_opt.step()
                # train generator again
                generator_opt.zero_grad()
                fake_target = torch.logical_xor(labels, torch.tensor(1)).float()
                generator_loss = process_batch(encoded_batch, fake_target, stego_analyser, criterion)
                scale_gradients(generator, 1 - alpha_balance)
                generator_opt.step()

                # generator_losses.append(generator_loss)
                # discriminator_losses_on_real.append(real_loss)
                # discriminator_losses_on_fake.append(fake_loss)

                # run callbacks
                for callback in callbacks:
                    callback(epoch)

            losses = {'Generator': np.mean(generator_losses),
                      'Discriminator on fake': np.mean(discriminator_losses_on_fake),
                      'Discriminator on real': np.mean(discriminator_losses_on_real)
                      }
            logger.policies(losses, epoch)


def main():
    pass
