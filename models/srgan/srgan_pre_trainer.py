import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers

from models.vgg.vgg import VggBuilder
from models.common.losses import compute_perceptual_loss, compute_pixel_loss, compute_discriminator_loss
from models.common.metrics import compute_psnr, compute_ssim


class SrganPreTrainer:
    """A helper class for training an SRGAN model."""

    def __init__(self, generator, learning_rate=1e-4, rgb_mean=np.array([0.4488, 0.4371, 0.4040]) * 255):
        """Constructor.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            learning_rate: The learning rate.
        """
        
        self.rgb_mean = rgb_mean

        self.generator_optimizer = optimizers.Adam(learning_rate)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              generator=generator,
                                              generator_optimizer=self.generator_optimizer)

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

        for _ in range(epochs_to_run):
            current_epoch = checkpoint.step.numpy() // steps
            performed_steps = steps * current_epoch

            self.__log(f'epoch: {current_epoch + 1}/{epochs}')

            average_loss = 0

            for low_res_img, high_res_img in dataset.take(steps):
                current_step = checkpoint.step.numpy()

                loss, psnr, ssim, lr = self.__train_step(
                    low_res_img, high_res_img)
                
                average_loss += loss

                if not np.any(performed_steps):
                    current_step_in_set = current_step + 1
                else:
                    current_step_in_set = current_step % performed_steps + 1

                self.__log(
                    f'step: {current_step_in_set:3.0f}/{steps:3.0f}, completed: {current_step_in_set / steps * 100:3.0f}%, loss: {loss.numpy():8.6f}, psnr: {psnr.numpy():5.2f}, ssim: {ssim.numpy():4.2f}, lr: {lr.numpy():.10f}', indent_level=1, end='\n', flush=True)

                checkpoint.step.assign_add(1)

            if current_epoch > 0 and (current_epoch + 1) % 10 == 0:
                checkpoint_manager.save()

            if current_epoch == epochs - 1:
                checkpoint_manager.save()

            self.__log(
                f'done: average loss: {average_loss / steps:.2f}', indent_level=1, end='\n', flush=True)
            self.__log('')

    def restore(self):
        """Restores the latest checkpoint if it exists."""

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint)
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

        checkpoint = self.checkpoint

        generator = checkpoint.generator
        optimizer = checkpoint.generator_optimizer

        with tf.GradientTape() as gen_tape:
            # low_res_img = tf.cast(low_res_img, tf.float32)
            # high_res_img = tf.cast(high_res_img, tf.float32)

            super_res_img = generator(low_res_img, training=True)
            loss = compute_pixel_loss(high_res_img, super_res_img)

        gen_vars = generator.trainable_variables
        gen_grads = gen_tape.gradient(loss, gen_vars)
        gen_mapped_grads = zip(gen_grads, gen_vars)

        optimizer.apply_gradients(gen_mapped_grads)
        
        denorm_hr = (high_res_img * 127.5) + self.rgb_mean
        denorm_sr = (super_res_img * 127.5) + self.rgb_mean

        psnr = compute_psnr(denorm_hr, denorm_sr)
        ssim = compute_ssim(denorm_hr, denorm_sr)

        # psnr = compute_psnr(high_res_img, super_res_img)
        # ssim = compute_ssim(high_res_img, super_res_img)

        return loss, psnr, ssim, optimizer.lr

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
