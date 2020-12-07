import os
import argparse
import numpy as np

from tqdm import tqdm
from dpipe.io import save_numpy
from typing import Sequence, Iterator
from torch.optim import Adam, Optimizer
from dpipe.train.logging import TBLogger

from sgan.utils import *
from sgan.modules import *
from sgan.data import CelebDataset, TextLoader, BatchIterator
from sgan.stegonagraphy import SigmoidTorchEncoder, generate_random_key, bytes_to_bits


def run_experiment(*, device, download: bool, n_epoch: int, batch_size: int, n_noise_channels: int
                   , data_path: str, experiment_path: str, dcgan_experiment_path: str
                   , sgan_experiment_path:str, generator_load_epoch: int, embedding_fidelity: float):
    # path to save everything related to experiment
    data_path = Path(data_path).expanduser()
    experiment_path = Path(experiment_path).expanduser()
    # dataset and batch iterator
    dataset = CelebDataset(root=data_path, download=download)
    indices = list(range(len(dataset)))
    train_iterator = BatchIterator(dataset, indices, batch_size=batch_size)
    # text loader and encoder
    encoder = SigmoidTorchEncoder(beta=embedding_fidelity)
    text_loader = TextLoader()
    text_iterator = text_loader.create_generator()
    # stegoanalyser: distinguish image with message from just image
    stegoanalyser = Discriminator().to(device)
    stegoanalyser.apply(init_weights)
    # trained generators: from dcgan and sgan
    generator_dcgan = Generator().to(device)
    generator_dcgan.load_state_dict(torch.load(dcgan_experiment_path
                                                / f'generator/generator_{generator_load_epoch}'))
    generator_sgan = Generator().to(device)
    generator_sgan.load_state_dict(torch.load(sgan_experiment_path
                                              / f'generator/generator_{generator_load_epoch}'))

    optimizer_parameters = dict(lr=1e-4, betas=(0.5, 0.99))
    stegoanalyser_opt = Adam(stegoanalyser.parameters(), **optimizer_parameters)

    fixed_noise = torch.randn(64, n_noise_channels, 1, 1, device=device)

    def save_models(epoch):
        os.makedirs(experiment_path / f'stegoanalyser', exist_ok=True)
        save_torch(stegoanalyser, experiment_path / f'stegoanalyser/stegoanalyser_{epoch}')

    logger = TBLogger(experiment_path / 'logs')
    epoch_callbacks = [save_models]

    # train on real images: with/without messages
    train_on_real_img(
        stegoanalyser=stegoanalyser,
        train_iterator=train_iterator,
        text_iterator=text_iterator,
        device=device,
        n_epoch=n_epoch,
        stegoanalyser_opt=stegoanalyser_opt,
        callbacks=epoch_callbacks,
        logger=logger,
        encoder=encoder
    )
    # train on generator_dcgan synth images: with/without messages
    # train on generator_sgan synth images: with/withput messages

def train_on_real_img(*, stegoanalyser: nn.Module, train_iterator: BatchIterator, text_iterator: Iterator
                      , device: str, n_epoch: int, stegoanalyser_opt: Optimizer
                      , callbacks: Sequence[Callable] = None, logger: TBLogger
                      , encoder: SigmoidTorchEncoder):
    stegoanalyser = stegoanalyser.to(device)
    criterion = F.binary_cross_entropy_with_logits

    callbacks = callbacks or []
    for epoch in tqdm(range(n_epoch)):
        stegoanalyser_losses = []

        with train_iterator as iterator:
            for real_batch, _ in iterator:
                batch_size = len(real_batch)

                labels = np.random.choice([0, 1], (batch_size, 1, 1, 1))
                encoded_images = []
                for image, label in zip(real_batch, labels):
                    if label == 1:
                        msg = bytes_to_bits(next(text_iterator))
                        key = generate_random_key(image.shape[1:], len(msg))
                        image = encoder.encode(image, msg, key)
                    encoded_images.append(image)

                encoded_images = torch.stack(encoded_images)
                labels = torch.from_numpy(labels).float()
                #train stegoanalyzer
                stegoanalyser_opt.zero_grad()
                stegoanalyser_losses.append(
                    process_batch(encoded_images.detach(), labels, stegoanalyser, criterion))
                stegoanalyser_opt.step()

            #run callbacks
            for callbacks in callbacks:
                callbacks(epoch)

            losses = {'Stegoanalyser loss': np.mean(stegoanalyser_losses)}
            logger.policies(losses, epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--dcgan_experiment_path', type=str, required=True)
    parser.add_argument('--sgan_experiment_path', type=str, required=True)
    parser.add_argument('--generator_load_epoch', default=30, type=int)
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_epoch', default=30, type=int)
    parser.add_argument('--n_noise_channels', default=100, type=int)
    parser.add_argument('--data_path', default='~/celeba', type=str)
    parser.add_argument('--embedding_fidelity', default=10000, type=float)

    args = parser.parse_args()
    run_experiment(**vars(args))


if __name__ == '__main__':
    main()