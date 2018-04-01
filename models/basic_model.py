# """
#
# References:
#     - https://github.com/adeshpande3/Generative-Adversarial-Networks/blob/master/Generative%20Adversarial%20Networks%20Tutorial.ipynb
#
# Paper:
#     - https://arxiv.org/abs/1406.2661
# """
#
# import numpy as np
# import tensorflow as tf
# from tensorflow.contrib import layers
# from tensorflow.contrib import signal
# from tqdm import tqdm
#
# from asariri.dataset.features.asariri_features import GANFeature
# from nlp.text_classification.tc_utils.tc_config import ModelConfigBase
# from sarvam.helpers.print_helper import *
# from speech_recognition.sr_config.sr_config import *
# from tensorflow.contrib.learn import ModeKeys
# from tensorflow.python.training import session_run_hook
# import collections
# from tensorflow.python.training import training_util
# # from tensorflow.contrib.gan.estimator.GANEstimator
#
# class BasicModelConfig(ModelConfigBase):
#     def __init__(self, model_dir, batch_size):
#         self._model_dir = model_dir
#
#         self._z_dimensions = 16000
#         self._seed = 2018
#         self._batch_size = batch_size
#         self._keep_prob = 0.5
#         self._learning_rate = 1e-3
#         self._clip_gradients = 15.0
#         self._use_batch_norm = True
#         self._num_classes = len(POSSIBLE_COMMANDS) + 2
#
#     @staticmethod
#     def user_config(batch_size):
#         _model_dir = "experiments/asariri/models/BasicModel/"
#         config = BasicModelConfig(_model_dir, batch_size)
#         BasicModelConfig.dump(_model_dir, config)
#         return config
#
#
# class RunTrainOpsHook(session_run_hook.SessionRunHook):
#   """A hook to run train ops a fixed number of times."""
#
#   def __init__(self, train_ops, train_steps):
#     """Run train ops a certain number of times.
#
#     Args:
#       train_ops: A train op or iterable of train ops to run.
#       train_steps: The number of times to run the op(s).
#     """
#     # if not isinstance(train_ops, (list, tuple)):
#     #   train_ops = [train_ops]
#     self._train_ops = train_ops
#     self._train_steps = train_steps
#
#   def before_run(self, run_context):
#     # for i in range(self._train_steps):
#     #     print_info("$$$$$$$> {}".format(i))
#     #     print_info("RunTrainOpsHook :  {}".format(self._train_ops))
#         run_context.session.run(self._train_ops)
#
#
# class GANTrainSteps(
#     collections.namedtuple('GANTrainSteps', (
#         'generator_train_steps',
#         'discriminator_train_steps'
#     ))):
#   """Contains configuration for the GAN Training.
#
#   Args:
#     generator_train_steps: Number of generator steps to take in each GAN step.
#     discriminator_train_steps: Number of discriminator steps to take in each GAN
#       step.
#   """
#
#
#
# class BasicModel(tf.estimator.Estimator):
#     def __init__(self,
#                  asariri_config, run_config):
#         super(BasicModel, self).__init__(
#             model_fn=self._model_fn,
#             model_dir=asariri_config._model_dir,
#             config=run_config)
#
#         self.asariri_config = asariri_config
#
#
#
#         self._feature_type = GANFeature
#
#     def get_sequential_train_hooks(self, generator_train_op,
#                                    discriminator_train_op,
#                                    train_steps=GANTrainSteps(1, 1)):
#         """Returns a hooks function for sequential GAN training.
#
#         Args:
#           train_steps: A `GANTrainSteps` tuple that determines how many generator
#             and discriminator training steps to take.
#
#         Returns:
#           A function that takes a GANTrainOps tuple and returns a list of hooks.
#         """
#
#         def get_hooks():
#             generator_hook = RunTrainOpsHook(generator_train_op,
#                                              train_steps.generator_train_steps)
#             discriminator_hook = RunTrainOpsHook(discriminator_train_op,
#                                                  train_steps.discriminator_train_steps)
#             return [generator_hook, discriminator_hook]
#
#         return get_hooks()
#
#
#     def conv2d(self, x, W):
#         tf.logging.info("=========> {}".format(x))
#         tf.logging.info("=========> {}".format(W))
#         return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')
#
#     def avg_pool_2x2(self, x):
#         return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#     def discriminator(self, x_image, reuse=False):
#
#         print_info("Discriminator in action ")
#
#
#         with tf.variable_scope('discriminator') as scope:
#             if (reuse):
#                 tf.get_variable_scope().reuse_variables()
#             # First Conv and Pool Layers
#             W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
#             b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
#             h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
#             h_pool1 = self.avg_pool_2x2(h_conv1)
#
#             # Second Conv and Pool Layers
#             W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
#             b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
#             h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
#             h_pool2 = self.avg_pool_2x2(h_conv2)
#
#             # First Fully Connected Layer
#             W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32],
#                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
#             b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
#             h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
#             h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#             # Second Fully Connected Layer
#             W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
#             b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))
#
#             # Final Layer
#             y_conv = (tf.matmul(h_fc1, W_fc2) + b_fc2)
#         return y_conv
#
#     def generator(self, z, batch_size, z_dim, reuse=False):
#         with tf.variable_scope('generator') as scope:
#             if (reuse):
#                 tf.get_variable_scope().reuse_variables()
#             g_dim = 64  # Number of filters of first layer of generator
#             c_dim = 1  # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
#             s = 28  # Output size of the image
#             s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(
#                 s / 16)  # We want to slowly upscale the image, so these values will help
#             # make that change gradual.
#
#             h0 = tf.reshape(z, [batch_size, s16 + 1, s16 + 1, 25])
#             h0 = tf.nn.relu(h0)
#             # Dimensions of h0 = batch_size x 2 x 2 x 25
#
#             # First DeConv Layer
#             output1_shape = [batch_size, s8, s8, g_dim * 4]
#             W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#             b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1],
#                                              padding='SAME')
#             H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True,
#                                                    scope="g_bn1")
#             H_conv1 = tf.nn.relu(H_conv1)
#             # Dimensions of H_conv1 = batch_size x 3 x 3 x 256
#
#             # Second DeConv Layer
#             output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
#             W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#             b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1],
#                                              padding='SAME')
#             H_conv2 = tf.contrib.layers.batch_norm(inputs=H_conv2, center=True, scale=True, is_training=True,
#                                                    scope="g_bn2")
#             H_conv2 = tf.nn.relu(H_conv2)
#             # Dimensions of H_conv2 = batch_size x 6 x 6 x 128
#
#             # Third DeConv Layer
#             output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
#             W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#             b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1],
#                                              padding='SAME')
#             H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True,
#                                                    scope="g_bn3")
#             H_conv3 = tf.nn.relu(H_conv3)
#             # Dimensions of H_conv3 = batch_size x 12 x 12 x 64
#
#             # Fourth DeConv Layer
#             output4_shape = [batch_size, s, s, c_dim]
#             W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#             b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1],
#                                              padding='VALID')
#             H_conv4 = tf.nn.tanh(H_conv4)
#             # Dimensions of H_conv4 = batch_size x 28 x 28 x 1
#
#         return H_conv4
#
#     def generator_audio(self, z, batch_size, z_dim, reuse=False):
#         """
#         Same as generator above, here audio data is used in place of noise
#         :param z:
#         :param batch_size:
#         :param z_dim:
#         :param reuse:
#         :return:
#         """
#
#         print_info("Generator in action! Reuse : {} ".format(reuse))
#
#         print_error(tf.trainable_variables())
#
#         with tf.variable_scope('generator', reuse=reuse):
#             if (reuse):
#                 tf.get_variable_scope().reuse_variables()
#             g_dim = 64  # Number of filters of first layer of generator
#             c_dim = 1  # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
#             s = 28  # Output size of the image
#             s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(
#                 s / 16)  # We want to slowly upscale the image, so these values will help
#             # make that change gradual.
#
#             h0 = tf.reshape(z, [batch_size, s16 + 1, s16 + 1, 32*5])#980+25])
#             h0 = tf.nn.relu(h0)
#             # Dimensions of h0 = batch_size x 2 x 2 x 980
#
#             # First DeConv Layer
#             output1_shape = [batch_size, s8, s8, g_dim * 4]
#             W_conv1 = tf.get_variable('g_wconv1',
#                                       [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#
#             b_conv1 = tf.get_variable('g_bconv1',
#                                       [output1_shape[-1]],
#                                       initializer=tf.constant_initializer(.1))
#
#             H_conv1 = tf.nn.conv2d_transpose(h0,
#                                              W_conv1,
#                                              output_shape=output1_shape,
#                                              strides=[1, 2, 2, 1],
#                                              padding='SAME')
#
#             H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1,
#                                                    center=True,
#                                                    scale=True,
#                                                    is_training=True,
#                                                    scope="g_bn1")
#             H_conv1 = tf.nn.relu(H_conv1)
#             # Dimensions of H_conv1 = batch_size x 3 x 3 x 256
#
#             # Second DeConv Layer
#             output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
#             W_conv2 = tf.get_variable('g_wconv2',
#                                       [5, 5, output2_shape[-1],
#                                        int(H_conv1.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#
#             b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1],
#                                              padding='SAME')
#             H_conv2 = tf.contrib.layers.batch_norm(inputs=H_conv2, center=True, scale=True, is_training=True,
#                                                    scope="g_bn2")
#             H_conv2 = tf.nn.relu(H_conv2)
#             # Dimensions of H_conv2 = batch_size x 6 x 6 x 128
#
#             # Third DeConv Layer
#             output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
#             W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#             b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1],
#                                              padding='SAME')
#             H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True,
#                                                    scope="g_bn3")
#             H_conv3 = tf.nn.relu(H_conv3)
#             # Dimensions of H_conv3 = batch_size x 12 x 12 x 64
#
#             # Fourth DeConv Layer
#             output4_shape = [batch_size, s, s, c_dim]
#             W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
#                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
#             b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
#             H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1],
#                                              padding='VALID')
#             H_conv4 = tf.nn.tanh(H_conv4)
#             # Dimensions of H_conv4 = batch_size x 28 x 28 x 1
#
#         return H_conv4
#
#     def _model_fn(self, features, labels, mode, params):
#         """
#
#         :param features: of type `asariri.dataset.features.asariri_features.AudioImageFeature`.
#                         Expect Audio to be an flatten array of size 3920 and image size of 28 X 28,
#         :param labels:
#         :param mode:
#         :param params:
#         :return:
#         """
#
#         sample_image = None
#         training_hooks = None
#
#         # Create global step increment op.
#         self.global_step = training_util.get_or_create_global_step()
#         self.global_step_inc = self.global_step.assign_add(1)
#
#         z_placeholder = features[self._feature_type.AUDIO_OR_NOISE]  # Audio/Noise Placeholder to the discriminator
#         z_placeholder = tf.cast(z_placeholder, tf.float32)
#
#         tf.logging.info("=========> {}".format(z_placeholder))
#
#
#         if mode != ModeKeys.INFER:
#
#             x_placeholder =  features[self._feature_type.IMAGE]   # Placeholder for input image vectors to the generator
#
#             x_placeholder = tf.cast(x_placeholder, tf.float32)
#             tf.logging.info("=========> {}".format(x_placeholder))
#
#             Dx = self.discriminator(x_placeholder)  # Dx will hold discriminator outputs (unnormalized) for the real MNIST images
#
#             Gz = self.generator_audio(z_placeholder,
#                                       self.asariri_config._batch_size,
#                                       self.asariri_config._z_dimensions,
#                                       reuse=False)  # Gz holds the generated images
#
#             Dg = self.discriminator(Gz, reuse=True)  # Dg will hold discriminator outputs (unnormalized) for generated images
#
#             g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
#
#             d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
#             d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
#             d_loss = d_loss_real + d_loss_fake
#
#             tvars = tf.trainable_variables()
#             d_vars = [var for var in tvars if 'd_' in var.name]
#             g_vars = [var for var in tvars if 'g_' in var.name]
#
#         else:
#             sample_image = self.generator_audio(z_placeholder,
#                                       1,
#                                       self.asariri_config._z_dimensions,
#                                       reuse=False)  # Gz holds the generated images
#
#
#         # Loss, training and eval operations are not needed during inference.
#         loss = None
#         train_op = None
#         eval_metric_ops = {}
#
#         if mode != ModeKeys.INFER:
#             loss = g_loss #Lets observe only one of the loss
#             tf.summary.scalar(name= "g_loss", tensor=g_loss)
#             tf.summary.scalar(name= "d_loss", tensor=d_loss)
#
#             with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#                 trainerD = tf.train.AdamOptimizer(name="d_optimizer").minimize(d_loss, var_list=d_vars)
#                 trainerG = tf.train.AdamOptimizer(name="g_optimizer").minimize(g_loss, var_list=g_vars)
#                 training_hooks = self.get_sequential_train_hooks(trainerG, trainerD)
#
#         return tf.estimator.EstimatorSpec(
#             mode=mode,
#             predictions=sample_image,
#             loss=loss,
#             train_op=self.global_step_inc,
#             eval_metric_ops=eval_metric_ops,
#             training_hooks=training_hooks
#         )
#
# """
# python asariri/commands/run_experiments.py \
# --mode=train \
# --dataset-name=crawled_dataset \
# --data-iterator-name=crawled_data_iterator \
# --model-name=basic_model \
# --batch-size=32 \
# --num-epochs=50
#
# python asariri/commands/run_experiments.py \
# --mode=predict \
# --dataset-name=crawled_dataset \
# --data-iterator-name=crawled_data_iterator \
# --model-name=basic_model \
# --batch-size=32 \
# --num-epochs=5 \
# --model-dir=experiments/asariri/models/BasicModel/
# """
#
# """
# python asariri/commands/run_experiments.py \
# --mode=train \
# --dataset-name=crawled_data \
# --data-iterator-name=raw_data_iterators \
# --model-name=basic_model \
# --batch-size=32 \
# --num-epochs=5
#
# python asariri/commands/run_experiments.py \
# --mode=predict \
# --dataset-name=crawled_data \
# --data-iterator-name=raw_data_iterators \
# --model-name=basic_model \
# --batch-size=32 \
# --num-epochs=5 \
# --model-dir=experiments/asariri/models/BasicModel/
# """