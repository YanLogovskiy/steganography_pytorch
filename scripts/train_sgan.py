import os
import argparse

from tqdm import tqdm
from torch.nn import functional as F
from typing import Sequence, Iterator
from torch.optim import Adam, Optimizer

from sgan.utils import *
from sgan.loggers import TBLogger
from sgan.modules import Generator, Discriminator
from sgan.data import CelebDataset, TextLoader, BatchIterator
from sgan.stegonagraphy import SigmoidTorchEncoder, generate_random_key, bytes_to_bits


def run_experiment(*, device, download: bool, data_path: str, experiment_path: str, n_epoch: int, batch_size: int,
                   n_noise_channels: int, embedding_fidelity: float, loss_balancer: float):
    # path to save everything related to experiment
    data_path = Path(data_path).expanduser()
    experiment_path = Path(experiment_path).expanduser()

    # dataset and image iterator
    dataset = CelebDataset(root=data_path, download=download)
    indices = list(range(len(dataset)))
    train_iterator = BatchIterator(dataset, indices, batch_size=batch_size)
    # text loader and encoder
    encoder = SigmoidTorchEncoder(beta=embedding_fidelity)
    text_loader = TextLoader()
    text_iterator = text_loader.create_generator()
    # create models (both discriminator models have similar structure)
    image_analyser = Discriminator().to(device)
    message_analyzer = Discriminator().to(device)
    generator = Generator(in_channels=n_noise_channels).to(device)

    optimizer_parameters = dict(lr=2e-4, betas=(0.5, 0.99))
    generator_opt = Adam(generator.parameters(), **optimizer_parameters)
    image_analyser_opt = Adam(image_analyser.parameters(), **optimizer_parameters)
    message_analyser_opt = Adam(message_analyzer.parameters(), **optimizer_parameters)

    fixed_noise = torch.randn(64, n_noise_channels, 1, 1, device=device)

    def predict_on_fixed_noise(epoch, prefix='fixed_noise', compression=1):
        predict = to_numpy(inference_step(fixed_noise, generator))
        os.makedirs(experiment_path / prefix, exist_ok=True)
        save_numpy(predict, experiment_path / prefix / f'{epoch}.npy.gz', compression=compression)

    def save_models(epoch):
        save_torch(generator, experiment_path / 'generator')
        save_torch(image_analyser, experiment_path / 'discriminator')
        save_torch(message_analyzer, experiment_path / 'stego_analyser')

    logger = TBLogger(experiment_path / 'logs')
    epoch_callbacks = [predict_on_fixed_noise, save_models]

    train_sgan(
        generator=generator,
        image_analyser=image_analyser,
        message_analyser=message_analyzer,
        generator_opt=generator_opt,
        image_analyser_opt=image_analyser_opt,
        message_analyser_opt=message_analyser_opt,
        encoder=encoder,
        image_iterator=train_iterator,
        text_iterator=text_iterator,
        n_epoch=n_epoch,
        n_noise_channels=n_noise_channels,
        loss_balancer=loss_balancer,
        logger=logger,
        callbacks=epoch_callbacks,
        device=device
    )


def train_sgan(*, generator: nn.Module, image_analyser: nn.Module, message_analyser: nn.Module,
               generator_opt: Optimizer, image_analyser_opt: Optimizer, message_analyser_opt: Optimizer,
               encoder: SigmoidTorchEncoder, image_iterator: BatchIterator, text_iterator: Iterator, device: str,
               n_epoch: int, n_noise_channels: int, loss_balancer: float, callbacks: Sequence[Callable],
               logger: TBLogger):
    generator = generator.to(device)
    image_analyser = image_analyser.to(device)
    message_analyser = message_analyser.to(device)
    criterion = F.binary_cross_entropy_with_logits

    assert 0. < loss_balancer < 1., loss_balancer
    callbacks = callbacks or []
    for epoch in tqdm(range(n_epoch)):
        generator_image_losses = []
        generator_message_losses = []
        image_analyser_real_losses = []
        image_analyser_fake_losses = []
        message_analyser_losses = []

        with image_iterator as iterator:
            for real_batch, _ in iterator:
                batch_size = len(real_batch)
                image_analyser_opt.zero_grad()
                # train discriminator on real
                real_images_target = torch.ones(batch_size, 1, 1, 1)
                image_analyser_real_losses.append(
                    process_batch(real_batch, real_images_target, image_analyser, criterion))
                # train discriminator on fake
                generated_images = generator(generate_noise(batch_size, n_noise_channels, device))
                image_analyser_fake_losses.append(
                    process_batch(generated_images.detach(), torch.zeros(batch_size, 1, 1, 1),
                                  image_analyser, criterion))
                image_analyser_opt.step()
                # train generator
                generator_opt.zero_grad()
                generator_image_losses.append(process_batch(generated_images, torch.ones(batch_size, 1, 1, 1),
                                                            image_analyser, criterion))
                # rescale gradients
                scale_gradients(generator, loss_balancer)
                generator_opt.step()
                # TODO: start after several epochs?
                # start second part
                containers = generator(generate_noise(batch_size, n_noise_channels, device))
                labels = np.random.choice([0, 1], (batch_size, 1, 1, 1))
                encoded_images = []
                for container, label in zip(containers, labels):
                    label = np.random.choice([0, 1])
                    if label == 1:
                        msg = bytes_to_bits(next(text_iterator))
                        key = generate_random_key(container.shape[1:], len(msg))
                        container = encoder.encode(container, msg, key)
                    encoded_images.append(container)

                encoded_images = torch.stack(encoded_images)
                labels = torch.from_numpy(labels).float()
                # train analyser
                message_analyser_opt.zero_grad()
                message_analyser_losses.append(
                    process_batch(encoded_images.detach(), labels, message_analyser, criterion))
                message_analyser_opt.step()
                # train generator again
                labels = torch.logical_xor(labels, torch.tensor(1)).float()
                generator_opt.zero_grad()
                generator_message_losses.append(process_batch(encoded_images, labels, message_analyser, criterion))
                scale_gradients(generator, 1 - loss_balancer)
                generator_opt.step()

            # run callbacks
            for callback in callbacks:
                callback(epoch)

            losses = {'Generator image': np.mean(generator_image_losses),
                      'Generator message': np.mean(generator_message_losses),
                      'Image discriminator on real': np.mean(image_analyser_real_losses),
                      'Image discriminator on fake': np.mean(image_analyser_fake_losses),
                      'Message discriminator': np.mean(message_analyser_losses)
                      }
            logger.policies(losses, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_epoch', default=25, type=int)
    parser.add_argument('--n_noise_channels', default=64, type=int)
    parser.add_argument('--loss_balancer', default=0.85, type=float)
    parser.add_argument('--embedding_fidelity', default=10, type=float)
    parser.add_argument('--data_path', default='~/celeba', type=str)

    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == '__main__':
    main()
