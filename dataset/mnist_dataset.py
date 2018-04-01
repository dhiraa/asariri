import os
from overrides import overrides
from asariri.dataset.dataset_interface import IDataset
from sarvam.helpers.downloaders import *
from sarvam.helpers.print_helper import *
import matplotlib.pyplot as plt

class Mnist(IDataset):
    """
    Downloads, mnist data set and creates three buckets based of hash of filenames
    """

    def __init__(self, audio_folder, image_folder, is_live):
        IDataset.__init__(self, audio_folder, image_folder, is_live)
        self.set_num_channels(1)
        self.set_name("Mnist")
        self.is_live = is_live



    @overrides
    def preprocess(self):
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        hash_code = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
        extract_path = os.path.join(self._images_dir, 'mnist_all_images')
        save_path = os.path.join(self._images_dir, 'train-images-idx3-ubyte.gz')

        if os.path.exists(extract_path):
            print_info('Found MNIST dataset')
        else:
            if not os.path.exists(self._images_dir):
                os.makedirs(self._images_dir)

            if not os.path.exists(save_path):
                with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading minist}') as pbar:
                    urlretrieve(
                        url,
                        save_path,
                        pbar.hook)

            assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
                '{} file is corrupted.  Remove the file and try again.'.format(save_path)

            os.makedirs(extract_path)
            try:
                ungzip(save_path, extract_path, "all_images", self._images_dir)
            except Exception as err:
                shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
                raise err

                # Remove compressed data

            os.remove(save_path)

        final_Data_path = extract_path
        for filename in get_dir_content(final_Data_path):
            if filename.endswith(".jpg"):
                set_name = which_set(filename=filename, validation_percentage=10, testing_percentage=10)
                if set_name == "training":
                    self._train_files.append(filename)
                elif set_name == "validation":
                    self._val_files.append(filename)
                elif set_name == "testing":
                    self._test_files.append(filename)


    def get_train_files(self):
        return self._train_files

    def get_val_files(self):
        return self._val_files

    def get_test_files(self):
        return self._test_files[:10]

    def predict_on_test_files(self, data_iterator, estimator):

        predictions_fn = estimator.predict(input_fn=data_iterator.get_test_input_function(),
                                           hooks=[])

        predictions = []

        for r in predictions_fn:
            images = r
            predictions.append(images)
            my_i = images.squeeze()
            plt.imshow(my_i, cmap="gray_r")
            plt.pause(1)
        input("press enter to exit!")

#
# mm = Minist("../data/asariri_mnist_images/")
# mm.preprocess()
