"""
Paper   : https://arxiv.org/abs/1411.1784
Git     : https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/CGAN.py
        : https://github.com/carpedm20/DCGAN-tensorflow
"""

import collections

from PIL import Image
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

from asariri.dataset.features.asariri_features import GANFeature
from asariri.models.utils.ops import *
from nlp.text_classification.tc_utils.tc_config import ModelConfigBase
from sarvam.helpers.print_helper import *


class CGANConfig(ModelConfigBase):
    def __init__(self, model_dir, batch_size):
        self._model_dir = model_dir

        self.learning_rate = 0.001
        self.alpha = 0.15
        self.beta1 = 0.4
        self.z_dim = 30
        
        self.batch_size = batch_size

    @staticmethod
    def user_config(batch_size):
        _model_dir = "experiments/asariri/minist_iterator/models/VanillaGAN/"
        config = CGANConfig(_model_dir, batch_size)
        CGANConfig.dump(_model_dir, config)
        return config


def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
        images[:save_size * save_size],
        (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im


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

        if global_step % 2 == 0:
            samples = run_context.session.run(self._z_image)
            channel = self._z_image.get_shape()[-1]
            if channel == 1:
                images_grid = images_square_grid(samples, "L")
            else:
                images_grid = images_square_grid(samples, "RGB")

            images_grid.save('/tmp/asariri_{}.png'.format(global_step))
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


class CGAN(tf.estimator.Estimator):
    def __init__(self,
                 gan_config, run_config):
        super(CGAN, self).__init__(
            model_fn=self._model_fn,
            model_dir=gan_config._model_dir,
            config=run_config)

        self.gan_config = gan_config

        self._feature_type = GANFeature

    def discriminator(self, x, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("discriminator", reuse=reuse):
            # merge image and label
            y = tf.reshape(y, [self.gan_config.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(x, y)

            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'),
                           is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.gan_config.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'),
                           is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, y, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        with tf.variable_scope("generator", reuse=reuse):
            # merge noise and label
            z = concat([z, y], 1)

            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'),
                                is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'),
                                is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.gan_config.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.gan_config.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'),
                   is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.gan_config.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

        return out

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
--batch-size=32 \
--num-epochs=2

python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=mnist_dataset \
--data-iterator-name=mnist_iterator \
--model-name=vanilla_gan \
--batch-size=32 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/VanillaGAN/
"""

"""
CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--model-name=vanilla_gan \
--batch-size=32 \
--num-epochs=2

python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_dataset \
--data-iterator-name=crawled_data_iterator \
--model-name=vanilla_gan \
--batch-size=32 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/VanillaGAN/
"""

"""
CUDA_VISIBLE_DEVICES=0 python asariri/commands/run_experiments.py \
--mode=train \
--dataset-name=crawled_dataset_v1 \
--data-iterator-name=crawled_data_iterator \
--model-name=vanilla_gan \
--batch-size=32 \
--num-epochs=100

python asariri/commands/run_experiments.py \
--mode=predict \
--dataset-name=crawled_dataset_v1 \
--data-iterator-name=crawled_data_iterator \
--model-name=vanilla_gan \
--batch-size=32 \
--num-epochs=2 \
--model-dir=experiments/asariri/models/VanillaGAN/
"""
