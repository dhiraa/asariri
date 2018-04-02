import os
from overrides import overrides
from asariri.dataset.dataset_interface import IDataset
from asariri.helpers.downloaders import *
from asariri.helpers.print_helper import *
import matplotlib.pyplot as plt
from asariri.utils.deprecated import *

class Cifar10(IDataset):
    """
    Downloads, Cifar10 data set and creates three buckets based of hash of filenames
    """

    def __init__(self, audio_folder, image_folder, is_live):
        IDataset.__init__(self, audio_folder, image_folder, is_live)
        self.set_name("Cifar10")
        self.is_live = is_live

        if "_bw_" in image_folder:
            self.set_num_channels(1)
        else:
            self.set_num_channels(3)



    @overrides
    def preprocess(self):
        for filename in get_dir_content(self._images_dir):
            if filename.endswith(".png"):
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
            plt.imshow(images)
            plt.pause(1)
        input("press enter to exit!")

#
# mm = Minist("../data/asariri_Cifar10_images/")
# mm.preprocess()
