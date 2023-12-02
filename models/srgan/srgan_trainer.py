import numpy as np
import tensorflow as tf

from datetime import datetime

from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from models.vgg.vgg import VggBuilder


class SrganTrainer:
    """A helper class for training an SRGAN model."""

    def __init__(self, generator, discriminator, learning_rate=1e-4):
        """Constructor.

        Args:
            generator: The generator model.
            discriminator: The discriminator model.
            learning_rate: The learning rate.
        """

        self.vgg = VggBuilder(layer='block5_conv4').build(
            input_shape=(None, None, 3))

        self.mean_absolute_error = tf.keras.losses.MeanAbsoluteError()
        self.mean_squared_error = tf.keras.losses.MeanSquaredError()
        self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=False)

        self.generator_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                        psnr=tf.Variable(-1.0),
                                                        optimizer=optimizers.Adam(
            learning_rate),
            model=generator)

        self.discriminator_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                                            psnr=tf.Variable(-1.0),
                                                            optimizer=optimizers.Adam(
            learning_rate),
            model=generator)

        # now = datetime.now()
        # formatted_date = now.strftime('%Y-%m-%d-%H-%M-%S')

        self.generator_checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.generator_checkpoint,
                                                                       directory='./.cache/checkpoints/srgan/generator/',
                                                                       max_to_keep=1000)

        self.discriminator_checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.generator_checkpoint,
                                                                           directory='./.cache/checkpoints/srgan/discriminator/',
                                                                           max_to_keep=1000)

        self.restore()

    def train(self, dataset, epochs, steps):
        """Trains the model.

        Args:
            dataset: The training dataset.
            epochs: The number of epochs.
            steps: The number of steps per epoch.

        """

        generator_checkpoint = self.generator_checkpoint
        generator_checkpoint_manager = self.generator_checkpoint_manager

        discriminator_checkpoint = self.discriminator_checkpoint
        discriminator_checkpoint_manager = self.discriminator_checkpoint_manager

        if generator_checkpoint.step.numpy() != discriminator_checkpoint.step.numpy():
            raise ValueError(
                'Generator and discriminator steps are not equal.')

        performed_steps = generator_checkpoint.step.numpy()
        performed_epochs = performed_steps // steps
        epochs_to_run = epochs - performed_epochs

        if performed_steps > 0:
            self.__log(f'resuming from epoch: {performed_epochs + 1}/{epochs}')
            self.__log(f'epochs to run: {epochs_to_run}')

        for _ in range(epochs_to_run):
            current_epoch = generator_checkpoint.step.numpy() // steps
            performed_steps = steps * current_epoch

            self.__log(f'epoch: {current_epoch + 1}/{epochs}')

            for low_res_img, high_res_img in dataset.take(steps):
                current_step = generator_checkpoint.step.numpy()

                perceptual_loss, discriminator_loss = self.__train_step(
                    low_res_img, high_res_img)

                if not np.any(performed_steps):
                    current_step_in_set = current_step + 1
                else:
                    current_step_in_set = current_step % performed_steps + 1

                self.__log(
                    f'step: {current_step_in_set}/{steps}, completed: {current_step_in_set / steps * 100:.0f}%, perceptual loss: {perceptual_loss.numpy():.2f}, discriminator loss {discriminator_loss.numpy():.2f}', indent_level=1, end='\n', flush=True)

                generator_checkpoint.step.assign_add(1)
                discriminator_checkpoint.step.assign_add(1)

            generator_checkpoint_manager.save()
            discriminator_checkpoint_manager.save()

            self.__log('')

    def restore(self):
        """Restores the latest checkpoint if it exists."""

        if self.generator_checkpoint_manager.latest_checkpoint:
            self.generator_checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint)
            print(
                f'generator restored at step: {self.checkpoint.step.numpy()}.')

        if self.discriminator_checkpoint_manager.latest_checkpoint:
            self.discriminator_checkpoint.restore(
                self.discriminator_checkpoint_manager.latest_checkpoint)
            print(
                f'discriminator restored at step: {self.discriminator_checkpoint.step.numpy()}.')

    def __train_step(self, low_res_img, high_res_img):
        """Performs a single training step.

        Args:
            low_res_img: The low resolution image.
            high_res_img: The high resolution image.

        Returns:
            The loss.
        """

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            low_res_img = tf.cast(low_res_img, tf.float32)
            high_res_img = tf.cast(high_res_img, tf.float32)

            prediction = self.generator_checkpoint.model(
                low_res_img, training=True)

            disc_hr_out = self.discriminator_checkpoint.model(
                high_res_img, training=True)
            disc_sr_out = self.discriminator_checkpoint.model(
                prediction, training=True)

            content_loss = self.__content_loss(high_res_img, prediction)
            generator_loss = self.__generator_loss(disc_sr_out)

            perceptual_loss = content_loss + generator_loss * 0.001
            disc_loss = self.__discriminator_loss(disc_hr_out, disc_sr_out)

        generator_variables = self.generator_checkpoint.model.trainable_variables
        discriminator_variables = self.discriminator_checkpoint.model.trainable_variables

        generator_gradients = generator_tape.gradient(
            perceptual_loss, generator_variables)
        discriminator_gradients = discriminator_tape.gradient(
            disc_loss, discriminator_variables)

        generator_mapped_gradients = zip(
            generator_gradients, generator_variables)
        discriminator_mapped_gradients = zip(
            discriminator_gradients, discriminator_variables)

        self.generator_checkpoint.optimizer.apply_gradients(
            generator_mapped_gradients)
        self.discriminator_checkpoint.optimizer.apply_gradients(
            discriminator_mapped_gradients)

        return perceptual_loss, disc_loss

    @tf.function
    def __content_loss(self, high_res_img, super_res_img):
        """Calculates the content loss of the super resolution image using the keras VGG model.

        Args:
            high_res_img: The high resolution image.
            super_res_img: The generated super resolution image.

        Returns:
            The content loss.
        """

        high_res_img = preprocess_input(high_res_img)
        super_res_img = preprocess_input(super_res_img)

        high_res_features = self.vgg(high_res_img) / 12.75
        super_res_features = self.vgg(super_res_img) / 12.75

        loss = self.mean_squared_error(high_res_features, super_res_features)
        return loss

    @tf.function
    def __generator_loss(self, super_res_img):
        return self.binary_cross_entropy(tf.ones_like(super_res_img), super_res_img)

    @tf.function
    def __discriminator_loss(self, high_res_img, super_res_img):
        high_res_loss = self.binary_cross_entropy(
            tf.ones_like(high_res_img), high_res_img)
        super_res_loss = self.binary_cross_entropy(
            tf.zeros_like(super_res_img), super_res_img)

        return high_res_loss + super_res_loss

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
