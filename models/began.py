"""
Paper:
    - https://arxiv.org/abs/1703.10717
Git:
    - https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/BEGAN.py
    - https://github.com/carpedm20/BEGAN-tensorflow
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

from asariri.models.utils.ops import *
from asariri.models.utils.utils import *
from asariri.asariri_utils.images.image import *


class BEGANConfig(ModelConfigBase):
    def __init__(self, model_dir,
                 batch_size,
                 num_image_channels,
                 image_size):
        self._model_dir = model_dir

        self.num_image_channels = num_image_channels
        self.image_size = int(image_size)


        self.learning_rate = 0.0002
        self.batch_size = batch_size

        # BEGAN Parameter
        self.gamma = 0.75
        self.lamda = 0.001

        self.beta1 = 0.5
        self.alpha = 0.15

    @staticmethod
    def user_config(batch_size, data_iterator):
        _model_dir = EXPERIMENT_MODEL_ROOT_DIR + "/" + data_iterator.name + "/began/"
        config = BEGANConfig(_model_dir, batch_size, data_iterator.get_image_channels(),
                             data_iterator.get_image_size())
        BEGANConfig.dump(_model_dir, config)
        return config

class RunTrainOpsHook(session_run_hook.SessionRunHook):
    """A hook to run train ops a fixed number of times."""

    def __init__(self, train_op, train_steps):

        self._train_op = train_op
        self._train_steps = train_steps

    def before_run(self, run_context):
        for _ in range(self._train_steps):
            run_context.session.run(self._train_op)

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

class UserLogHook(session_run_hook.SessionRunHook):
    def __init__(self, z_image, d_loss, g_loss, global_Step):
        self._z_image = z_image
        self._d_loss = d_loss
        self._g_loss = g_loss
        self._global_Step = global_Step


    def before_run(self, run_context):
        global_step = run_context.session.run(self._global_Step)

        print_info("global_step {}".format(global_step))

        if global_step % 2 == 0:
            samples = run_context.session.run(self._z_image)
            channel = self._z_image.get_shape()[-1]
            if channel == 1:
                images_grid= images_square_grid(samples, "L")
            else:
                images_grid= images_square_grid(samples, "RGB")

            if not os.path.exists(EXPERIMENT_DATA_ROOT_DIR+'/began/' ): os.makedirs(EXPERIMENT_DATA_ROOT_DIR+'/began/' )

            images_grid.save(EXPERIMENT_DATA_ROOT_DIR+'/began/'  + '/asariri_{}.png'.format(global_step))

        if global_step % 2 == 0:
            dloss, gloss = run_context.session.run([self._d_loss, self._g_loss])
            print_info("\nDiscriminator Loss: {:.4f}... Generator Loss: {:.4f}".format(dloss, gloss))


class BEGAN(tf.estimator.Estimator):
    def __init__(self,
                 gan_config,
                 run_config):
        super(BEGAN, self).__init__(
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


    def discriminator(self, x, out_channel_dim, is_training=True, reuse=False):
        # It must be Auto-Encoder style architecture
        # Architecture : (64)4c2s-FC32_BR-FC64*14*14_BR-(1)4dc2s_S
        with tf.variable_scope("discriminator", reuse=reuse):
            # net = tf.nn.relu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = tf.layers.conv2d(x, 64, 4, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   name='d_conv1')
            net = tf.nn.relu(net)

            tf.logging.info("======> net: {}".format(net))
            print_error("net1: {} ".format(net))

            size = (self.gan_config.image_size//2)


            net = tf.reshape(net, [self.gan_config.batch_size, size*size*64])

            # code = tf.nn.relu(bn(linear(net, 32, scope='d_fc6'), is_training=is_training, scope='d_bn6'))
            code = tf.contrib.layers.fully_connected(inputs=net, num_outputs=32, scope="d_fc6")
            code = tf.contrib.layers.batch_norm(code,
                                         decay=0.9,
                                         updates_collections=None,
                                         epsilon=1e-5,
                                         scale=True,
                                         is_training=is_training,
                                         scope='d_bn6')
            code = tf.nn.relu(code)

            print_error("code: {} ".format(code))
            # net = tf.nn.relu(bn(linear(code, 64 * 14 * 14, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            size = (self.gan_config.image_size // 2)
            net =  tf.contrib.layers.fully_connected(inputs=code, num_outputs=64 * size * size, scope="d_fc3")

            net = tf.contrib.layers.batch_norm(net,
                                         decay=0.9,
                                         updates_collections=None,
                                         epsilon=1e-5,
                                         scale=True,
                                         is_training=is_training,
                                         scope='d_bn3')
            print_error("net: {} ".format(net))
            print_error(net)

            size = (self.gan_config.image_size // 2)
            net = tf.reshape(net, [self.gan_config.batch_size, size, size, 64])
            print_error(net)

            # out = tf.nn.sigmoid(deconv2d(net, [self.gan_config.batch_size, 28, 28, 1], 4, 4, 2, 2, name='d_dc5'))
            net =  tf.layers.conv2d_transpose(net, out_channel_dim, 4, strides=2, padding='same', name='d_dc5')
            out = tf.nn.sigmoid(net)

            print_info("==================================")
            print_info(out)
            print_info(x)
            # recon loss
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / self.gan_config.batch_size
            print_info("==================================")
            print_error(recon_error)

            return out, recon_error, code

    # def generator(self, z, is_training=True, reuse=False):
    #     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    #     # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    #     with tf.variable_scope("generator", reuse=reuse):
    #         net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
    #         net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
    #         net = tf.reshape(net, [self.gan_config.batch_size, 7, 7, 128])
    #         net = tf.nn.relu(
    #             bn(deconv2d(net, [self.gan_config.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
    #                scope='g_bn3'))
    #
    #         out = tf.nn.sigmoid(deconv2d(net, [self.gan_config.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))
    #
    #     return out

    def generator(self, z, out_channel_dim, is_training=True, reuse=False):
        """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_training: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """

        # with tf.variable_scope('generator', reuse=not is_training):
        #     # First fully connected layer
        #     x1 = tf.layers.dense(z, 7 * 7 * 512 * 2)
        #     # Reshape it to start the convolutional stack
        #     x1 = tf.reshape(x1, (-1, 7, 7, 512 * 2))
        #     #         x1 = tf.layers.batch_normalization(x1, training=training)
        #     x1 = tf.maximum(self.gan_config.alpha * x1, x1)
        #     # 7x7x512 now
        #     #         print(x1)
        #     x2 = tf.layers.conv2d_transpose(x1, 256 * 2, 5, strides=1, padding='same')
        #     x2 = tf.layers.batch_normalization(x2, training=is_training)
        #     x2 = tf.maximum(self.gan_config.alpha * x2, x2)
        #     # 7x7x256 now
        #     #         print(x2)
        #     x3 = tf.layers.conv2d_transpose(x2, 128 * 2, 5, strides=2, padding='same')
        #     x3 = tf.layers.batch_normalization(x3, training=is_training)
        #     x3 = tf.maximum(self.gan_config.alpha * x3, x3)
        #     # 14x14x128 now
        #     #         print(x3)
        #
        #     x4 = tf.layers.conv2d_transpose(x3, 64 * 2, 5, strides=2, padding='same')
        #     x4 = tf.layers.batch_normalization(x4, training=is_training)
        #     x4 = tf.maximum(self.gan_config.alpha * x4, x4)
        #
        #     # Output layer
        #     logits = tf.layers.conv2d_transpose(x4, out_channel_dim, 5, strides=1, padding='same')
        #     # 28x28x3 now
        #     #         print(logits)3
        #     out = tf.tanh(logits)
        #
        #     print_info("======>out: {}".format(out))
        #
        #     return out

        with tf.variable_scope('generator', reuse=not is_training):
            filter_size = 512

            # First fully connected layer
            x = tf.layers.dense(z, 8 * 8 * filter_size)
            # Reshape it to start the convolutional stack
            x = tf.reshape(x, (-1, 8, 8, filter_size))
            x = tf.maximum(self.gan_config.alpha * x, x)

            x = tf.layers.conv2d_transpose(x, filter_size // 2, 5, strides=1, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.maximum(self.gan_config.alpha * x, x)

            filter_size = filter_size // 4
            # 32 //  8 = srt(4)  => 2 => (8) -> 16 -> 32
            # 64 //  8 = srt(8)  => 3 => (8) -> 16 -> 32 -> 64
            # 128 // 8 = srt(16) => 4 => (8) -> 16 -> 32 -> 64 -> 128

            for i in range(int(math.sqrt(self.gan_config.image_size // 8))):
                filter_size = filter_size // 2
                x = tf.layers.conv2d_transpose(x, filter_size, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.maximum(self.gan_config.alpha * x, x)

                print_info("======>out: {}".format(x))

            # Output layer
            logits = tf.layers.conv2d_transpose(x, out_channel_dim, 5, strides=1, padding='same')
            # 28x28x3 now
            #         print(logits)3
            out = tf.tanh(logits)

            print_info("======>out: {}".format(out))

            return out

    def model_loss(self, input_real, input_z, out_channel_dim, global_step, is_training):
        """
        Get the loss for the discriminator and generator
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """

        print_error(input_real)
        print_error(input_z)

        # output of D for real images
        D_real_img, D_real_err, D_real_code = self.discriminator(input_real, out_channel_dim=out_channel_dim,is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(input_z,  out_channel_dim=out_channel_dim, is_training=True, reuse=False)
        D_fake_img, D_fake_err, D_fake_code = self.discriminator(G, out_channel_dim=out_channel_dim, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss = D_real_err - self.k * D_fake_err

        # get loss for generator
        g_loss = D_fake_err

        # convergence metric
        M = D_real_err + tf.abs(self.gan_config.gamma * D_real_err - D_fake_err)

        # operation for updating k
        update_k = self.k.assign(
            tf.clip_by_value(self.k + self.gan_config.lamda * (self.gan_config.gamma * D_real_err - D_fake_err), 0, 1))

        print_hooks = UserLogHook(G, d_loss, g_loss, global_step)

        return d_loss, g_loss, print_hooks, update_k

    def model_opt(self, d_loss, g_loss, global_step):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        #
        # g_vars = [var for var in t_vars if 'g_' in var.name]

        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_optim = tf.train.AdamOptimizer(self.gan_config.learning_rate,
                                             beta1=self.gan_config.beta1) \
                .minimize(d_loss, var_list=d_vars,global_step=global_step)
            g_optim = tf.train.AdamOptimizer(self.gan_config.learning_rate * 5, beta1=self.gan_config.beta1) \
                .minimize(g_loss, var_list=g_vars,global_step=global_step)

        # tf.logging.info("=========> {}".format(d_train_opt))
        # tf.logging.info("=========> {}".format(g_train_opt))

        return d_optim, g_optim

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

        is_training = mode != ModeKeys.INFER

        # Create global step increment op.
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_inc = self.global_step.assign_add(0)

        """ BEGAN variable """
        self.k = tf.Variable(0., trainable=False)

        z_placeholder = features[self._feature_type.AUDIO_OR_NOISE]  # Audio/Noise Placeholder to the discriminator
        tf.logging.info("=========> {}".format(z_placeholder))

        z_placeholder = tf.cast(z_placeholder, tf.float32, name="z_placeholder")

        tf.logging.info("=========> {}".format(z_placeholder))

        if is_training:

            x_placeholder = features[self._feature_type.IMAGE]  # Placeholder for input image vectors to the generator
            tf.logging.info("=========> {}".format(x_placeholder))

            x_placeholder = tf.cast(x_placeholder, tf.float32, name="x_placeholder")
            tf.logging.info("=========> {}".format(x_placeholder))

            num_img_channel = x_placeholder.get_shape()[-1]

            print_error(num_img_channel)
            d_loss, g_loss, print_hooks, update_k = self.model_loss(x_placeholder,
                                             z_placeholder,
                                             num_img_channel,
                                             self.global_step,
                                             is_training)

            d_train_opt, g_train_opt = self.model_opt(d_loss,
                                                      g_loss,
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
            update_k_hook = RunTrainOpsHook(update_k, 1)
            training_hooks.append(print_hooks)
            training_hooks.append(update_k_hook)

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
--dataset-name=crawled_dataset \
--image-folde=Images_bw_28x28 \
--data-iterator-name=crawled_data_iterator \
--model-name=began \
--batch-size=32 \
--num-epochs=2

CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--model-name=began \
--batch-size=32 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/crawleddataiterator/VanillaGAN/
"""
