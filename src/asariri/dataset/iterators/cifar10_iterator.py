from tqdm import tqdm
from asariri.helpers.print_helper import *
import numpy as np
from scipy.io import wavfile
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
import librosa
from PIL import Image
from asariri.dataset.features.asariri_features import GANFeature
from asariri.dataset.crawled_dataset import CrawledData
from asariri.utils.deprecated import *

class Cifar10DataIterator:
    """

    """

    def __init__(self, batch_size, num_epochs, dataset):
        self.name = "mnistdataiterator"
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._dataset = dataset
        self._feature_type = GANFeature

        self.image_mode = 'L'
        self.image_channels = 3
        self.noise_dim = 30


    def get_image_size(self):
        return 32

    def get_image_channels(self):
        return self.image_channels

    def data_generator(self, data, batch_size, mode='train'):

        print_info("Total number of files for {} is =====> {}".format(mode, len(data)))

        def generator():

            steps = 0
            IMAGE_MAX_VALUE = 255

            if mode == 'train':
                np.random.shuffle(data)

            batched_data_len = (len(data) // batch_size) * batch_size

            data_new = data[:batched_data_len]

            for i in tqdm(range(len(data_new)), desc=mode):

                if i % batch_size == 0:
                    steps += 1
                    # print_info("Steps: {}".format(steps))

                image_file_name = data_new[i]

                if (i + 1 == batched_data_len):
                    raise StopIteration  # Hack for the estimator from overshooting the iterator

                try:
                    image_data = Image.open(image_file_name)
                    image_data = np.array(image_data).astype(float)

                    if len(image_data.shape) == 2:  # if the image size 2 it syas it is BW 2Dim change it to 3Dim
                        image_data = np.expand_dims(image_data, axis=2)

                    #Normalize the data
                    image_data = image_data / IMAGE_MAX_VALUE - 0.5
                    image_data = image_data * 2

                    noise = np.random.uniform(-1, 1, size=(self.noise_dim))
                except Exception as err:
                    print_error(str(err))

                yield {self._feature_type.IMAGE: image_data,
                       self._feature_type.AUDIO_OR_NOISE: noise}

        return generator

    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.data_generator(self._dataset.get_train_files(), self._batch_size, 'train'),
            target_key=None,  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return train_input_fn

    def get_val_input_fn(self):
        val_input_fn = generator_input_fn(
            x=self.data_generator(self._dataset.get_val_files(), self._batch_size, 'val'),
            target_key=None,
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return val_input_fn

    def get_test_input_function(self):
        val_input_fn = generator_input_fn(
            x=self.data_generator(self._dataset.get_test_files(), 1, 'test'),
            target_key=None,
            batch_size=1,
            shuffle=False,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return val_input_fn

# preprocessor = CrawledData("../data/asariri/")
# iter = RawDataIterator(8,5,preprocessor)
# generator = iter.data_generator(preprocessor.get_train_files())
#
# for res in generator():
#     pass
#
# generator = iter.data_generator(preprocessor.get_val_files())
#
# for res in generator():
#     pass

