import os
import argparse
import sklearn
import numpy as np

from tqdm import tqdm
from dpipe.io import save
from typing import Sequence, Iterator
from torch.optim import Adam, Optimizer
from dpipe.train.logging import TBLogger

from sgan.utils import *
from sgan.modules import *
from sgan.data import CelebDataset, TextLoader, DataBatchIterator, split_data
from sgan.stegonagraphy import SigmoidTorchEncoder, generate_random_key, bytes_to_bits


def run_on_real(*, device, download: bool, n_epoch: int, batch_size: int, data_path: str, experiment_path: str,
                embedding_fidelity: float):
    data_path = Path(data_path).expanduser()
    experiment_path = Path(experiment_path).expanduser()
    os.makedirs(experiment_path, exist_ok=True)
    # dataset and batch iterator: real images
    dataset = CelebDataset(root=data_path, download=download)
    train_indices, val_indices, test_indices = split_data(dataset, train_size=0.6, val_size=0.1, test_size=0.3)
    train_iterator = DataBatchIterator(dataset, train_indices, batch_size=batch_size)
    val_iterator = DataBatchIterator(dataset, val_indices, batch_size=batch_size)
    # save indices for reproducibility
    save(test_indices, experiment_path / 'test_indices.json')
    # text stuff
    encoder = SigmoidTorchEncoder(beta=embedding_fidelity)
    text_loader = TextLoader()
    text_iterator = text_loader.create_generator()
    # model
    stegoanalyser = Stegoanalyser().to(device)
    # stegoanalyser.apply(init_weights)
    optimizer_parameters = dict(lr=1e-4, betas=(0.9, 0.99))
    stegoanalyser_opt = Adam(stegoanalyser.parameters(), **optimizer_parameters)

    def save_models(epoch):
        os.makedirs(experiment_path / f'stegoanalyser', exist_ok=True)
        save_torch(stegoanalyser, experiment_path / f'stegoanalyser/stegoanalyser_{epoch}')

    logger = TBLogger(experiment_path / 'logs')
    epoch_callbacks = [save_models]
    # train on real images: with/without messages
    train_stego(
        stegoanalyser=stegoanalyser,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        text_iterator=text_iterator,
        n_epoch=n_epoch,
        stegoanalyser_opt=stegoanalyser_opt,
        callbacks=epoch_callbacks,
        logger=logger,
        encoder=encoder
    )


def train_stego(*, stegoanalyser: nn.Module,
                train_iterator: DataBatchIterator,
                val_iterator: DataBatchIterator,
                text_iterator: Iterator,
                n_epoch: int, stegoanalyser_opt: Optimizer,
                callbacks: Sequence[Callable] = None, logger: TBLogger,
                encoder: SigmoidTorchEncoder):
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
                        image = encoder.encode(transform_encoder(image), msg, key)
                        image = inverse_transform_encoder(image)
                    encoded_images.append(image)

                encoded_images = torch.stack(encoded_images)
                labels = torch.from_numpy(labels).float()
                # train stegoanalyzer
                stegoanalyser_opt.zero_grad()
                stegoanalyser_losses.append(
                    process_batch(encoded_images.detach(), labels, stegoanalyser, criterion))
                stegoanalyser_opt.step()

        with val_iterator as iterator:
            accuracy = []
            for real_batch, _ in iterator:
                batch_size = len(real_batch)

                labels = np.random.choice([0, 1], batch_size)
                encoded_images = []
                for image, label in zip(real_batch, labels):
                    if label == 1:
                        msg = bytes_to_bits(next(text_iterator))
                        key = generate_random_key(image.shape[1:], len(msg))
                        image = encoder.encode(transform_encoder(image), msg, key)
                        image = inverse_transform_encoder(image)
                    encoded_images.append(image)

                encoded_images = torch.stack(encoded_images)
                # evaluate stegoanalyzer
                out = inference_step(encoded_images, stegoanalyser).cpu().detach()
                out = torch.sigmoid(out) > 0.5
                out = out.reshape(len(encoded_images)).numpy()
                accuracy_score = sklearn.metrics.accuracy_score(labels, out)
                accuracy.append(accuracy_score)

            mean_accuracy = np.mean(accuracy)
            print(f'validation accuracy score {mean_accuracy}')

            losses = {'Stegoanalyser loss': np.mean(stegoanalyser_losses),
                      'Val accuracy': mean_accuracy}
            logger.policies(losses, epoch)

            # run callbacks
            for callback in callbacks:
                callback(epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epoch', default=15, type=int)
    parser.add_argument('--data_path', default='~/celeba', type=str)
    parser.add_argument('--embedding_fidelity', default=10000, type=float)

    args = parser.parse_args()
    run_on_real(**vars(args))


if __name__ == '__main__':
    main()
