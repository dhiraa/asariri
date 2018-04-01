"""
Reference: 
    - https://github.com/Mageswaran1989/deep-learning/blob/project_5/face_generation/
"""

import numpy as np
import tensorflow as tf
import math
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib import signal
from tqdm import tqdm

from asariri.dataset.features.asariri_features import GANFeature
from asariri.asariri_utils.asariri_config import *
from sarvam.helpers.print_helper import *
from speech_recognition.sr_config.sr_config import *
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.training import session_run_hook
import collections
from tensorflow.python.training import training_util
from matplotlib import pyplot
from asariri.asariri_utils.images.image import *
import math

class VanillaGANConfig(ModelConfigBase):
    def __init__(self, model_dir, batch_size, num_image_channels, image_size):
        self._model_dir = model_dir

        self.num_image_channels = num_image_channels
        self.image_size = image_size

        self.learning_rate = 0.001
        self.alpha = 0.15
        self.beta1 = 0.4
        self.z_dim = 30


    @staticmethod
    def user_config(batch_size, data_iterator):
        _model_dir = EXPERIMENT_MODEL_ROOT_DIR + "/" + data_iterator.name + "/vanilla_gan/"
        config = VanillaGANConfig(_model_dir, batch_size, data_iterator.get_image_channels(),
                                  data_iterator.get_image_size())
        VanillaGANConfig.dump(_model_dir, config)
        return config

class RunTrainOpsHook(session_run_hook.SessionRunHook):
    """A hook to run train ops a fixed number of times."""

    def __init__(self, train_op, train_steps):

        self._train_op = train_op
        self._train_steps = train_steps

    def before_run(self, run_context):
        for _ in range(self._train_steps):
            run_context.session.run(self._train_op)

class UserLogHook(session_run_hook.SessionRunHook):
    def __init__(self, z_image, d_loss, g_loss, global_Step):
        self._z_image = z_image
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._global_Step = global_Step


    def before_run(self, run_context):
        global_step = run_context.session.run(self._global_Step)

        print_info("global_step {}".format(global_step))

        if global_step % 2 == 0 or global_step % 3 == 0 :
            samples = run_context.session.run(self._z_image)
            channel = self._z_image.get_shape()[-1]
            if channel == 1:
                images_grid= images_square_grid(samples, "L")
            else:
                images_grid= images_square_grid(samples, "RGB")

            if not os.path.exists(EXPERIMENT_DATA_ROOT_DIR+'/vanilla_gan/' ): os.makedirs(EXPERIMENT_DATA_ROOT_DIR+'/vanilla_gan/' )

            images_grid.save(EXPERIMENT_DATA_ROOT_DIR+'/vanilla_gan/' + '/asariri_{}.png'.format(global_step))

        if global_step % 2 == 0:
            dloss, gloss = run_context.session.run([self._d_loss, self._g_loss])
            print_info("\nDiscriminator Loss: {:.4f}... Generator Loss: {:.4f}".format(dloss, gloss))


class GANTrainSteps(
    collections.namedtuple('GANTrainSteps', (
            'generator_train_steps',
            'discriminator_train_steps'
    ))):
    """Contains configuration for the GAN Training.
  
    Args:
      generator_train_steps: Number of generator steps to take in each GAN step.
      discriminator_train_steps: Number of discriminator steps to take in each GAN
        step.
    """

class VanillaGAN(tf.estimator.Estimator):
    def __init__(self,
                 gan_config, run_config):
        super(VanillaGAN, self).__init__(
            model_fn=self._model_fn,
            model_dir=gan_config._model_dir,
            config=run_config)

        self.gan_config = gan_config

        self._feature_type = GANFeature

    def get_sequential_train_hooks(self, generator_train_op,
                                   discriminator_train_op,
                                   train_steps=GANTrainSteps(1, 1)):
        """Returns a hooks function for sequential GAN training.

        Args:
          train_steps: A `GANTrainSteps` tuple that determines how many generator
            and discriminator training steps to take.

        Returns:
          A function that takes a GANTrainOps tuple and returns a list of hooks.
        """
        # print_info(generator_train_op)
        # print_info(discriminator_train_op)

        generator_hook = RunTrainOpsHook(generator_train_op,
                                         train_steps.generator_train_steps)
        discriminator_hook = RunTrainOpsHook(discriminator_train_op,
                                             train_steps.discriminator_train_steps)
        return [discriminator_hook, generator_hook]


    def discriminator(self, images, reuse=False):
        """
        Create the discriminator network
        :param image: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
        """

        with tf.variable_scope('discriminator', reuse=reuse):
            # Input layer is ?x28x28x3
            x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            relu1 = tf.maximum(0.02 * x1, x1)
            relu1 = tf.layers.dropout(relu1, rate=0.5)
            # 14x14x64
            #         print(x1)
            x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            bn2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = tf.maximum(0.02 * bn2, bn2)
            relu2 = tf.layers.dropout(relu2, rate=0.5)
            # 7x7x128
            #         print(x2)
            x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(0.02 * bn3, bn3)
            relu3 = tf.layers.dropout(relu3, rate=0.5)
            # 4x4x256
            #         print(x3)
            # Flatten it
            flat = tf.reshape(relu3, (-1, 4 * 4 * 256))
            logits = tf.layers.dense(flat, 1)
            #         print(logits)
            out = tf.sigmoid(logits)
            #         print('discriminator out: ', out)

            print_info("======>out: {}".format(out))

            return out, logits

    def generator(self, z, out_channel_dim, is_train=True):
        """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """

        with tf.variable_scope('generator', reuse=not is_train):
            filter_size = 512

            # First fully connected layer
            x = tf.layers.dense(z, 8 * 8 * filter_size)
            # Reshape it to start the convolutional stack
            x = tf.reshape(x, (-1, 8, 8, filter_size))
            x = tf.maximum(self.gan_config.alpha * x, x)

            x = tf.layers.conv2d_transpose(x, filter_size//2, 5, strides=1, padding='same')
            x = tf.layers.batch_normalization(x, training=is_train)
            x = tf.maximum(self.gan_config.alpha * x, x)

            filter_size = filter_size // 4
            # 32 //  8 = srt(4)  => 2 => (8) -> 16 -> 32
            # 64 //  8 = srt(8)  => 3 => (8) -> 16 -> 32 -> 64
            # 128 // 8 = srt(16) => 4 => (8) -> 16 -> 32 -> 64 -> 128

            for i in range(int(math.sqrt(self.gan_config.image_size // 8))):
                filter_size = filter_size // 2
                x = tf.layers.conv2d_transpose(x, filter_size, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=is_train)
                x = tf.maximum(self.gan_config.alpha * x, x)

                print_info("======>out: {}".format(x))

            # Output layer
            logits = tf.layers.conv2d_transpose(x, out_channel_dim, 5, strides=1, padding='same')
            # 28x28x3 now
            #         print(logits)3
            out = tf.tanh(logits)

            print_info("======>out: {}".format(out))
    
            return out

    def model_loss(self, input_real, input_z, out_channel_dim, global_step):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """
        #     print('Generator for fake images...')
        g_model = self.generator(input_z, out_channel_dim)
        #     print('Passing discriminator with real images...')
        d_model_real, d_logits_real = self.discriminator(input_real)
        #     print('Passing discriminator with fake images...')
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

        d_loss = d_loss_real + d_loss_fake

        print_hooks = UserLogHook(g_model, d_loss, g_loss, global_step)

        return d_loss, g_loss, print_hooks

    def model_opt(self, d_loss, g_loss, learning_rate, beta1, global_step):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """

        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # Optimize
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name="d_train_opt").\
            minimize(d_loss, var_list=d_vars, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        g_updates = [opt for opt in update_ops if opt.name.startswith('generator')]

        with tf.control_dependencies(g_updates):
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name="g_train_opt").\
                minimize(g_loss, var_list=g_vars, global_step=global_step)

        # tf.logging.info("=========> {}".format(d_train_opt))
        # tf.logging.info("=========> {}".format(g_train_opt))

        return d_train_opt, g_train_opt

    def _model_fn(self, features, labels, mode, params):
        """

        :param features: 
        :param labels: 
        :param mode: 
        :param params: 
        :return: 
        """

        sample_image = None
        training_hooks = None

        # Create global step increment op.
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_inc = self.global_step.assign_add(0)

        z_placeholder = features[self._feature_type.AUDIO_OR_NOISE]  # Audio/Noise Placeholder to the discriminator
        tf.logging.info("=========> {}".format(z_placeholder))

        z_placeholder = tf.cast(z_placeholder, tf.float32)

        tf.logging.info("=========> {}".format(z_placeholder))

        if mode != ModeKeys.INFER:

            x_placeholder = features[self._feature_type.IMAGE]  # Placeholder for input image vectors to the generator
            tf.logging.info("=========> {}".format(x_placeholder))

            x_placeholder = tf.cast(x_placeholder, tf.float32)
            tf.logging.info("=========> {}".format(x_placeholder))

            channel = x_placeholder.get_shape()[-1]
            d_loss, g_loss, print_hooks = self.model_loss(x_placeholder, z_placeholder, channel, self.global_step)

            d_train_opt, g_train_opt = self.model_opt(d_loss, g_loss,
                                                      self.gan_config.learning_rate,
                                                      self.gan_config.beta1,
                                                      self.global_step)
        else:
            sample_image = self.generator(z_placeholder, self.gan_config.num_image_channels)
            #changes are made to take image channels from data iterator just for prediction


        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            loss = g_loss + d_loss
            tf.summary.scalar(name="g_loss", tensor=g_loss)
            tf.summary.scalar(name="d_loss", tensor=d_loss)

            training_hooks = self.get_sequential_train_hooks(d_train_opt, g_train_opt)
            training_hooks.append(print_hooks)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=sample_image,
            loss=loss,
            train_op=self.global_step_inc,
            eval_metric_ops=eval_metric_ops,
            training_hooks=training_hooks
        )


"""
CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=mnist_dataset \
--data-iterator-name=mnist_iterator \
--model-name=vanilla_gan \
--image-folde=minist_bw_28x28 \
--batch-size=32 \
--num-epochs=2

python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=mnist_dataset \
--data-iterator-name=mnist_iterator \
--model-name=vanilla_gan \
--image-folde=minist_bw_28x28 \
--batch-size=32 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/mnistdataiterator/vanilla_gan/  \
--is-live=False
"""


"""
CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--image-folde=Images_bw_28x28 \
--model-name=vanilla_gan \
--batch-size=32 \
--num-epochs=100



CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--model-name=vanilla_gan \
--image-folde=Images_bw_28x28 \
--batch-size=32 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/crawleddataiterator/vanilla_gan/ \
--is-live=False
"""


# """
# #color and black and white uses same data iterator
# rm -rf experiments/asariri/models/crawleddataiterator/VanillaGAN/
#
# CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
# --mode=train \
# --dataset-name=crawled_dataset_v1 \
# --data-iterator-name=crawled_data_iterator \
# --model-name=vanilla_gan \
# --batch-size=32 \
# --num-epochs=100
#
# python asariri/commands/run_experiments.py \
# --mode=predict \
# --dataset-name=crawled_dataset_v1 \
# --data-iterator-name=crawled_data_iterator \
# --model-name=vanilla_gan \
# --batch-size=32 \
# --num-epochs=2 \
# --model-dir=experiments/asariri/models/mnistdataiterator/vanilla_gan/  \
# --is-live=False
# """



