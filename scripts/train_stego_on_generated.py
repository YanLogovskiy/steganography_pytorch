import os
import argparse
import numpy as np
import sklearn

from tqdm import tqdm
from typing import Sequence, Iterator
from torch.optim import Adam, Optimizer
from dpipe.train.logging import TBLogger

from sgan.utils import *
from sgan.modules import *
from sgan.data import TextLoader, ModelBatchGenerator
from sgan.stegonagraphy import SigmoidTorchEncoder, generate_random_key, bytes_to_bits


def run_on_generated(*, device, n_epoch: int, batch_size: int, batches_per_epoch, val_batches_per_epoch,
                     experiment_path: str, embedding_fidelity: float, model_path: str, n_noise_channels: int):
    model_path = Path(model_path).expanduser()
    experiment_path = Path(experiment_path).expanduser()

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path))
    train_iterator = ModelBatchGenerator(generator,
                                         batch_size=batch_size,
                                         batches_per_epoch=batches_per_epoch,
                                         n_noise_channels=n_noise_channels)
    val_iterator = ModelBatchGenerator(generator,
                                       batch_size=batch_size,
                                       batches_per_epoch=val_batches_per_epoch,
                                       n_noise_channels=n_noise_channels)
    # text stuff
    encoder = SigmoidTorchEncoder(beta=embedding_fidelity)
    text_loader = TextLoader()
    text_iterator = text_loader.create_generator()
    # model
    stegoanalyser = Stegoanalyser().to(device)
    stegoanalyser.apply(init_weights)
    optimizer_parameters = dict(lr=1e-4, betas=(0.5, 0.99))
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
                train_iterator: ModelBatchGenerator,
                val_iterator: ModelBatchGenerator,
                text_iterator: Iterator,
                n_epoch: int, stegoanalyser_opt: Optimizer,
                callbacks: Sequence[Callable] = None, logger: TBLogger,
                encoder: SigmoidTorchEncoder):
    criterion = F.binary_cross_entropy_with_logits
    callbacks = callbacks or []

    for epoch in tqdm(range(n_epoch)):
        stegoanalyser_losses = []
        with train_iterator as iterator:
            for batch in iterator:
                batch_size = len(batch)
                batch = inverse_transform_gan(batch)
                labels = np.random.choice([0, 1], (batch_size, 1, 1, 1))

                encoded_images = []
                for image, label in zip(batch, labels):
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

        # validation step
        with val_iterator as iterator:
            accuracy = []
            for batch in iterator:
                batch = inverse_transform_gan(batch)
                batch_size = len(batch)

                labels = np.random.choice([0, 1], batch_size)
                encoded_images = []
                for image, label in zip(batch, labels):
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

            mean_accuracy = np.mean(accuracy_score)
            print(f'validation accuracy score {mean_accuracy}')
            # run callbacks
            for callback in callbacks:
                callback(epoch)

            losses = {'Stegoanalyser loss': np.mean(stegoanalyser_losses),
                      'Val accuracy': mean_accuracy}
            logger.policies(losses, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--batches_per_epoch', default=1000)
    parser.add_argument('--val_batches_per_epoch', default=10)
    parser.add_argument('--n_epoch', default=15, type=int)
    parser.add_argument('--n_noise_channels', default=100, type=int)
    parser.add_argument('--embedding_fidelity', default=10000, type=float)

    args = parser.parse_args()
    run_on_generated(**vars(args))


if __name__ == '__main__':
    main()
