import os
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers

from models.srgan.data_processing import denormalize_output

from models.vgg.vgg import VggBuilder
from models.common.losses import compute_pixel_loss, compute_perceptual_loss, compute_generator_loss, compute_discriminator_loss
from models.common.metrics import compute_psnr, compute_ssim


class SrganTrainer:
    """A helper class for training an SRGAN model."""

    def __init__(self, generator, discriminator, generator_lr=1e-4, discriminator_lr=1e-4):
        """Constructor.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            learning_rate: The learning rate.
        """

        # self.vgg_layers = ['block1_conv2', 'block2_conv2',
        #                    'block3_conv4', 'block4_conv4', 'block5_conv4']
        # self.vgg_layer_weights = [0.03125, 0.0625, 0.125, 0.25, 0.5]

        self.vgg_layers = ['block5_conv4']
        self.vgg_layer_weights = [1.0]

        self.vgg = VggBuilder(layers=self.vgg_layers).build(
            input_shape=(None, None, 3))

        self.generator_optimizer = optimizers.Adam(generator_lr)
        self.discriminator_optimizer = optimizers.Adam(discriminator_lr)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator=generator,
                                              discriminator=discriminator,
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory='./.cache/checkpoints/srgan',
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

        now = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        csv_file = f'./.cache/logs/srgan/{now}.csv'

        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        with open(csv_file, 'w') as log_file:
            log_writer = csv.writer(log_file, delimiter=',')
            log_writer.writerow(['epoch', 'step', 'loss', 'loss_mse', 'loss_vgg',
                                'loss_g', 'loss_d', 'd_x_hr', 'd_x_sr', 'psnr', 'ssim'])

            for _ in range(epochs_to_run):
                current_epoch = checkpoint.step.numpy() // steps
                performed_steps = steps * current_epoch

                self.__log(f'epoch: {current_epoch + 1}/{epochs}')

                avg_loss = 0
                avg_loss_mse = 0
                avg_loss_vgg = 0
                avg_loss_g = 0
                avg_loss_d = 0

                avg_d_x_hr = 0
                avg_d_x_sr = 0

                avg_psnr = 0
                avg_ssim = 0

                for low_res_img, high_res_img in dataset.take(steps):
                    current_step = checkpoint.step.numpy()

                    loss, loss_mse, loss_vgg, loss_g, loss_d, d_x_hr, d_x_sr, psnr, ssim = self.__train_step(
                        low_res_img, high_res_img)

                    avg_loss += loss
                    avg_loss_mse += loss_mse
                    avg_loss_vgg += loss_vgg
                    avg_loss_g += loss_g
                    avg_loss_d += loss_d

                    avg_d_x_hr += d_x_hr
                    avg_d_x_sr += d_x_sr

                    avg_psnr += psnr
                    avg_ssim += ssim

                    if not np.any(performed_steps):
                        current_step_in_set = current_step + 1
                    else:
                        current_step_in_set = current_step % performed_steps + 1

                    log_writer.writerow([current_epoch + 1, current_step + 1, loss.numpy(), loss_mse.numpy(), loss_vgg.numpy(
                    ), loss_g.numpy(), loss_d.numpy(), d_x_hr.numpy(), d_x_sr.numpy(), psnr.numpy(), ssim.numpy()])
                    log_file.flush()

                    self.__log(
                        f'step: {current_step_in_set:3.0f}/{steps:3.0f}, completed: {current_step_in_set / steps * 100:3.0f}%, loss: {loss.numpy():8.6f}, loss_mse(g): {loss_mse.numpy():8.6f}, loss_vgg(g): {loss_vgg.numpy():8.6f}, loss_bce(g): {loss_g.numpy():6.4f}, loss_bce(d): {loss_d.numpy():6.4f}, d(x_hr): {d_x_hr.numpy():4.2f}, d(x_sr): {d_x_sr.numpy():4.2f}, psnr(y): {psnr.numpy():5.2f}, ssim(y): {ssim.numpy():4.2f}', indent_level=1, end='\n', flush=True)

                    checkpoint.step.assign_add(1)

                if current_epoch > 0 and (current_epoch + 1) % 10 == 0:
                    checkpoint_manager.save()

                avg_loss /= steps
                avg_loss_mse /= steps
                avg_loss_vgg /= steps
                avg_loss_g /= steps
                avg_loss_d /= steps

                avg_d_x_hr /= steps
                avg_d_x_sr /= steps

                avg_psnr /= steps
                avg_ssim /= steps

                self.__log('-' * 195, indent_level=1, end='\n', flush=True)
                self.__log(f'done: {"".rjust(25, " ")} loss: {avg_loss:8.6f}, loss_mse(g): {avg_loss_mse:8.6f}, loss_vgg(g): {avg_loss_vgg:8.6f}, loss_bce(g): {avg_loss_g:6.4f}, loss_bce(d): {avg_loss_d:6.4f}, d(x_hr): {avg_d_x_hr:4.2f}, d(x_sr): {avg_d_x_sr:4.2f}, psnr(y): {avg_psnr:5.2f}, ssim(y): {avg_ssim:4.2f}', indent_level=1, end='\n', flush=True)
                self.__log('')

    def restore(self):
        """Restores the latest checkpoint if it exists."""

        checkpoint = self.checkpoint
        manager = self.checkpoint_manager

        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print(f'model restored at step: {checkpoint.step.numpy()}.')

    @tf.function
    def __train_step(self, low_res_img, high_res_img):
        """Performs a single training step.

        Args:
            low_res_img: The low resolution image.
            high_res_img: The high resolution image.

        Returns:
            The loss.
        """

        checkpoint = self.checkpoint

        generator = checkpoint.generator
        discriminator = checkpoint.discriminator

        gen_opt = checkpoint.generator_optimizer
        disc_opt = checkpoint.discriminator_optimizer

        vgg = self.vgg
        vgg_layer_weights = self.vgg_layer_weights

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            super_res_img = generator(low_res_img, training=True)

            disc_pred_hr = discriminator(high_res_img, training=True)
            disc_pred_sr = discriminator(super_res_img, training=True)

            loss_d = compute_discriminator_loss(disc_pred_hr, disc_pred_sr)
            loss_g = compute_generator_loss(disc_pred_sr)

            loss_mse = compute_pixel_loss(high_res_img, super_res_img)
            loss_vgg = compute_perceptual_loss(
                high_res_img, super_res_img, vgg, vgg_layer_weights, feature_scale=1 / 12.75) / 500

            content_loss = loss_mse * 0.5 + loss_vgg * 0.5
            loss = content_loss + loss_g * 1e-3

        gen_vars = generator.trainable_variables
        disc_vars = discriminator.trainable_variables

        gen_grads = gen_tape.gradient(loss, gen_vars)
        disc_grads = disc_tape.gradient(loss_d, disc_vars)

        gen_mapped_grads = zip(gen_grads, gen_vars)
        disc_mapped_grads = zip(disc_grads, disc_vars)

        gen_opt.apply_gradients(gen_mapped_grads)
        disc_opt.apply_gradients(disc_mapped_grads)

        d_x_hr = tf.reduce_mean(disc_pred_hr)
        d_x_sr = tf.reduce_mean(disc_pred_sr)

        denorm_hr = denormalize_output(high_res_img)
        denorm_sr = denormalize_output(super_res_img)

        psnr = compute_psnr(denorm_hr, denorm_sr)
        ssim = compute_ssim(denorm_hr, denorm_sr)

        return loss, loss_mse, loss_vgg, loss_g, loss_d, d_x_hr, d_x_sr, psnr, ssim

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
