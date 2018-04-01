from tqdm import tqdm
from sarvam.helpers.print_helper import *
import numpy as np
from scipy.io import wavfile
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
import librosa
from PIL import Image
from asariri.dataset.features.asariri_features import GANFeature
from asariri.dataset.crawled_dataset import CrawledData

class CrawledDataIterator:
    """
    
    """

    def __init__(self, batch_size, num_epochs, dataset):
        self.name = "crawleddataiterator"

        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._dataset = dataset
        self._feature_type = GANFeature

    def get_image_size(self):
        return self._dataset.image_size

    def get_image_channels(self):
        return self._dataset.num_channels

    def melspectrogram(self, sample_rate, audio):
        # mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=98, n_fft=1024, hop_length=2048) #3920 samples
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=16, n_fft=1024, hop_length=2048) # 128samples
        mfcc = mfcc.astype(np.float32)
        # print_info(mfcc.flatten().shape)
        return mfcc.flatten()

    def data_generator(self, data, batch_size, mode='train'):

        print_info("Total number of files for {} is =====> {}".format(mode, len(data)))

        def generator():
            IMAGE_MAX_VALUE = 255

            # if mode == 'train': #We dont hawant more randomness than what we have!
            #     np.random.shuffle(data)

            batched_data_len = (len(data)//batch_size) * batch_size

            data_new = data[:batched_data_len]

            # print_error(data_new)

            for i in tqdm(range(len(data_new)), desc=mode):
                data_dict  =data_new[i]
                audio_file_name = data_dict["audio"]
                image_file_name = data_dict["image"]
                person_name = data_dict["label"]

                # if(i+1 == batched_data_len):
                if(i > batched_data_len):
                    print_error("Iterator Exhausted!")
                    raise StopIteration #Hack for the estimator from overshooting the iterator

                try:
                    sample_rate, wav = wavfile.read(audio_file_name)
                    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
                    wav = self.melspectrogram(sample_rate=sample_rate, audio=wav)

                    if wav.shape[0] != 128*5:
                        wav = np.pad(wav, (0, 128*5 - wav.shape[0]), mode="constant", constant_values=(0,0))

                    if mode != 'test':
                        image_data = Image.open(image_file_name)
                        image_data = np.array(image_data).astype(float)
                        image_data =  image_data/ IMAGE_MAX_VALUE - 0.5
                        image_data =  image_data *2

                        if len(image_data.shape) == 2:
                            image_data = np.expand_dims(image_data, axis=2)
                    else:
                        image_data = np.array("none")


                    # print_info("{} =====> {} {}".format(mode, audio_file_name, image_file_name))
                    # print_info("{} ===> {} {}".format(i, image_data.shape, wav.shape))

                    if(wav.shape[0] != 128*5):
                        raise RuntimeWarning("{} has problematic shape {}".format(audio_file_name, wav.shape))

                    noise = np.random.normal(-1, 1, [100])
                    wav = np.concatenate([noise, wav], axis=0)
                    # exit(-1)

                except Exception as err:
                    print_error(str(err))
                    exit()

                res = {self._feature_type.AUDIO_OR_NOISE: wav,
                 self._feature_type.IMAGE: image_data}

                yield res

        return generator

    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.data_generator(self._dataset.get_train_files(), self._batch_size, 'train'),
            target_key=None,  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size,
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
            queue_capacity=3 * self._batch_size,
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
            queue_capacity=3 * self._batch_size,
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

