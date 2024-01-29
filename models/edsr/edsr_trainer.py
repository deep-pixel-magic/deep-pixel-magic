import os
import csv

import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers

from models.vgg.vgg import VggBuilder
from models.common.metrics import compute_psnr, compute_ssim
from models.common.losses import compute_pixel_loss

from models.edsr.data_processing import denormalize_output


class EdsrNetworkTrainer:
    """A helper class for training an EDSR model."""

    def __init__(self, model, learning_rate=1e-4, use_content_loss=False):
        """Constructor.

        Args:
            model: The EDSR model.
            loss: The loss function.
            learning_rate: The learning rate.
        """

        optimizer = optimizers.Adam(learning_rate)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              model=model,
                                              optimizer=optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory='./.cache/checkpoints/edsr/',
                                                             max_to_keep=1000)

        self.restore()

    def train(self, dataset, epochs, steps):
        """Trains the model.

        Args:
            dataset: The training dataset.
            epochs: The number of epochs.
            steps: The number of steps per epoch.
        """

        checkpoint = self.checkpoint
        checkpoint_manager = self.checkpoint_manager

        performed_steps = checkpoint.step.numpy()
        performed_epochs = performed_steps // steps
        epochs_to_run = epochs - performed_epochs

        if performed_steps > 0:
            self.__log(f'epochs completed: {performed_epochs}/{epochs}')
            self.__log(f'epochs to run: {epochs_to_run}')

        csv_file = './.cache/logs/edsr.csv'
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        with open(csv_file, 'w') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')
            log_writer.writerow(['epoch', 'step', 'loss', 'psnr', 'ssim'])

            for _ in range(epochs_to_run):

                current_epoch = checkpoint.step.numpy() // steps
                performed_steps = steps * current_epoch

                self.__log(f'epoch: {current_epoch + 1}/{epochs}')

                avg_loss = 0
                avg_psnr = 0
                avg_ssim = 0

                for low_res_img, high_res_img in dataset.take(steps):

                    current_step = checkpoint.step.numpy()
                    current_loss, current_psnr, current_ssim = self.__train_step(
                        low_res_img, high_res_img)

                    avg_loss += current_loss
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim

                    if not np.any(performed_steps):
                        current_step_in_set = current_step + 1
                    else:
                        current_step_in_set = current_step % performed_steps + 1

                    log_writer.writerow(
                        [current_epoch + 1, current_step + 1, current_loss.numpy(), current_psnr.numpy(), current_ssim.numpy()])
                    log_file.flush()

                    self.__log(
                        f'step: {current_step_in_set:3.0f}/{steps:3.0f}, completed: {current_step_in_set / steps * 100:3.0f}%, mse(y): {current_loss.numpy():7.2f}, psnr(y): {current_psnr.numpy():5.2f}, ssim(y): {current_ssim.numpy():3.2f}', indent_level=1, end='\n', flush=True)

                    checkpoint.step.assign_add(1)

                checkpoint_manager.save()

                avg_loss /= steps
                avg_psnr /= steps
                avg_ssim /= steps

                self.__log('-' * 80, indent_level=1, end='\n', flush=True)

                self.__log(
                    f'done: mse(y): {avg_loss:6.4f}, psnr(y): {avg_psnr:5.2f}, ssim(y): {avg_ssim:5.2f}', indent_level=1, end='\n', flush=True)

                self.__log('')

    def restore(self):
        """Restores the latest checkpoint if it exists."""

        if self.checkpoint_manager.latest_checkpoint:

            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f'model restored at step: {self.checkpoint.step.numpy()}.')

    @tf.function
    def __train_step(self, low_res_img, high_res_img):
        """Performs a single training step.

        Args:
            low_res_img: The low resolution image.
            high_res_img: The high resolution image.

        Returns:
            The loss.
        """

        model = self.checkpoint.model
        optimizer = self.checkpoint.optimizer

        with tf.GradientTape() as tape:

            super_res_img = model(low_res_img, training=True)

            loss = compute_pixel_loss(high_res_img, super_res_img)

        variables = model.trainable_variables

        gradients = tape.gradient(loss, variables)

        mapped_gradients = zip(gradients, variables)
        optimizer.apply_gradients(mapped_gradients)

        denorm_hr = denormalize_output(high_res_img)
        denorm_sr = denormalize_output(super_res_img)

        psnr = compute_psnr(denorm_hr, denorm_sr)
        ssim = compute_ssim(denorm_hr, denorm_sr)

        return loss, psnr, ssim

    def __log(self, message, indent_level=0, end='\n', flush=False):
        """Prints the specified message to the console.

        Args:
            message: The message to be printed.
            indent_level: The indentation level.
            end: The string to be appended to the end of the message.
            flush: Specifies whether the output buffer should be flushed after printing the message.
        """

        prefix = " " * indent_level * 2
        print(prefix + message, end=end, flush=flush)
