import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers

from models.vgg.vgg import VggBuilder
from models.common.losses import compute_pixel_loss, compute_perceptual_loss, compute_generator_loss, compute_discriminator_loss
from models.common.metrics import compute_psnr, compute_ssim


class SrganTrainer:
    """A helper class for training an SRGAN model."""

    def __init__(self, generator, discriminator, generator_lr=1e-4, discriminator_lr=1e-4, rgb_mean=np.array([0.4488, 0.4371, 0.4040]) * 255):
        """Constructor.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            learning_rate: The learning rate.
        """

        self.rgb_mean = rgb_mean

        self.vgg_layers = ['block1_conv2', 'block2_conv2',
                           'block3_conv4', 'block4_conv4', 'block5_conv4']
        self.vgg_layer_weights = [0.03125, 0.0625, 0.125, 0.25, 0.5]

        # self.vgg_layers = ['block5_conv4']
        # self.vgg_layer_weights = [1]

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

        for _ in range(epochs_to_run):
            current_epoch = checkpoint.step.numpy() // steps
            performed_steps = steps * current_epoch

            self.__log(f'epoch: {current_epoch + 1}/{epochs}')

            for low_res_img, high_res_img in dataset.take(steps):
                current_step = checkpoint.step.numpy()

                loss, dxhr, dxsr, pixel_loss, perceptual_loss, gen_loss, disc_loss, psnr, ssim, gen_lr, disc_lr = self.__train_step(
                    low_res_img, high_res_img)

                if not np.any(performed_steps):
                    current_step_in_set = current_step + 1
                else:
                    current_step_in_set = current_step % performed_steps + 1

                self.__log(
                    f'step: {current_step_in_set:3.0f}/{steps:3.0f}, completed: {current_step_in_set / steps * 100:3.0f}%, loss: {loss.numpy():8.6f}, dhr(x): {dxhr.numpy():4.2f}, dsr(x): {dxsr.numpy():4.2f}, pixel loss: {pixel_loss.numpy():8.6f}, perceptual loss: {perceptual_loss.numpy():8.6f}, generator loss: {gen_loss.numpy():6.4f}, discriminator loss: {disc_loss.numpy():6.4f}, psnr: {psnr.numpy():5.2f}, ssim: {ssim.numpy():4.2f}, glr: {gen_lr.numpy():.10f}, dlr: {disc_lr.numpy():.7f}', indent_level=1, end='\n', flush=True)

                checkpoint.step.assign_add(1)

            if current_epoch > 0 and (current_epoch + 1) % 10 == 0:
                checkpoint_manager.save()

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
            # low_res_img = tf.cast(low_res_img, tf.float32)
            # high_res_img = tf.cast(high_res_img, tf.float32)

            super_res_img = generator(low_res_img, training=True)

            disc_out_hr = discriminator(high_res_img, training=True)
            disc_out_sr = discriminator(super_res_img, training=True)

            gen_loss = compute_generator_loss(disc_out_sr) * 1e-3
            disc_loss = compute_discriminator_loss(disc_out_hr, disc_out_sr)

            denorm_hr = (high_res_img * 127.5) + self.rgb_mean
            denorm_sr = (super_res_img * 127.5) + self.rgb_mean

            pixel_loss = compute_pixel_loss(high_res_img, super_res_img)
            perceptual_loss = compute_perceptual_loss(denorm_hr, denorm_sr, vgg, vgg_layer_weights, feature_scale=1 / 12.75) / 100000
            
            content_loss = pixel_loss * 0.5 + perceptual_loss * 0.5
            loss = content_loss + gen_loss

        gen_vars = generator.trainable_variables
        disc_vars = discriminator.trainable_variables

        gen_grads = gen_tape.gradient(loss, gen_vars)
        disc_grads = disc_tape.gradient(disc_loss, disc_vars)

        gen_mapped_grads = zip(gen_grads, gen_vars)
        disc_mapped_grads = zip(disc_grads, disc_vars)

        gen_opt.apply_gradients(gen_mapped_grads)
        disc_opt.apply_gradients(disc_mapped_grads)
        
        dxhr = tf.reduce_mean(disc_out_hr)
        dxsr = tf.reduce_mean(disc_out_sr)

        psnr = compute_psnr(denorm_hr, denorm_sr)
        ssim = compute_ssim(denorm_hr, denorm_sr)

        return loss, dxhr, dxsr, pixel_loss, perceptual_loss, gen_loss, disc_loss, psnr, ssim, gen_opt.lr, disc_opt.lr

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
