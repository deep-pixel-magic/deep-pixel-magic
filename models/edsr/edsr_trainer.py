import os
import csv

import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from models.vgg.vgg import VggBuilder
from models.common.metrics import compute_psnr, compute_ssim
from models.common.losses import compute_pixel_loss


class EdsrNetworkTrainer:
    """A helper class for training an EDSR model."""

    def __init__(self, model, learning_rate=1e-4):
        """Constructor.

        Args:
            model: The EDSR model.
            loss: The loss function.
            learning_rate: The learning rate.
        """

        self.vgg = VggBuilder(layers=['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']).build(
            input_shape=(None, None, 3))

        self.mean_absolute_error = tf.keras.losses.MeanAbsoluteError()
        self.mean_squared_error = tf.keras.losses.MeanSquaredError()

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=optimizers.Adam(
                                                  learning_rate),
                                              model=model)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory='./.cache/checkpoints/edsr/',
                                                             max_to_keep=200)

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

        with open(csv_file, 'a') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')
            log_writer.writerow(['epoch', 'step', 'loss', 'psnr', 'ssim'])

            for _ in range(epochs_to_run):

                current_epoch = checkpoint.step.numpy() // steps
                performed_steps = steps * current_epoch

                self.__log(f'epoch: {current_epoch + 1}/{epochs}')

                for low_res_img, high_res_img in dataset.take(steps):

                    current_step = checkpoint.step.numpy()
                    current_loss, current_psnr, current_ssim = self.__train_step(
                        low_res_img, high_res_img)

                    if not np.any(performed_steps):
                        current_step_in_set = current_step + 1
                    else:
                        current_step_in_set = current_step % performed_steps + 1

                    log_writer.writerow(
                        [current_epoch, checkpoint.step.numpy() + 1, current_loss.numpy(), current_psnr.numpy(), current_ssim.numpy()])
                    log_file.flush()

                    self.__log(
                        f'step: {current_step_in_set}/{steps}, completed: {current_step_in_set / steps * 100:.0f}%, loss: {current_loss.numpy():.2f}, psnr: {current_psnr.numpy():.2f}, ssim: {current_ssim.numpy():.2f}', indent_level=1, end='\n', flush=True)

                    checkpoint.step.assign_add(1)

                checkpoint_manager.save()
                self.__log('')

    def restore(self):
        """Restores the latest checkpoint if it exists."""

        if self.checkpoint_manager.latest_checkpoint:

            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f'model restored at step: {self.checkpoint.step.numpy()}.')

    def __train_step(self, low_res_img, high_res_img):
        """Performs a single training step.

        Args:
            low_res_img: The low resolution image.
            high_res_img: The high resolution image.

        Returns:
            The loss.
        """

        with tf.GradientTape() as tape:

            low_res_img = tf.cast(low_res_img, tf.float32)
            high_res_img = tf.cast(high_res_img, tf.float32)

            super_res_img = self.checkpoint.model(low_res_img, training=True)

            # perceptual_loss = self.__content_loss(high_res_img, super_res_img)
            pixel_loss = compute_pixel_loss(high_res_img, super_res_img)

            # loss = perceptual_loss + pixel_loss
            loss = pixel_loss

            psnr = compute_psnr(high_res_img, super_res_img)
            ssim = compute_ssim(high_res_img, super_res_img)

        variables = self.checkpoint.model.trainable_variables

        gradients = tape.gradient(loss, variables)
        mapped_gradients = zip(gradients, variables)

        self.checkpoint.optimizer.apply_gradients(mapped_gradients)

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
