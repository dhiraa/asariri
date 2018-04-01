import os
import os
from pprint import pformat
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import shuffle

from asariri.dataset.dataset_interface import IDataset
from sarvam.helpers.print_helper import *
from sarvam.helpers.downloaders import *

from asariri.asariri_utils.audio.recording import record_audio


class CrawledData(IDataset):
    def __init__(self, audio_folder, image_folder, is_live):
        IDataset.__init__(self, audio_folder, image_folder, is_live)

        if "_bw_" in image_folder:
            self.set_num_channels(1)
        else:
            self.set_num_channels(3)

        self.set_image_size(int(image_folder.split("x")[-1]))

        self.set_name("CrawledData")

    def is_audio_file(self, path):
        return path.split("/")[-3] == "audio"

    def is_image_file(self, path):
        return path.split("/")[-3] == self._images_dir.split("/")[-1]

    def preprocess(self):
        audio_files = []
        image_files = []

        audio_files_dict = defaultdict(list)
        image_files_dict = defaultdict(list)

        for path in get_dir_content(self._audio_dir):
            if self.is_audio_file(path):
                audio_files.append(path)

        for path in get_dir_content(self._images_dir):
            # print_debug(path)
            if self.is_image_file(path):
                image_files.append(path)


        for image_file_path in image_files:
            person = image_file_path.split("/")[-2]
            image_files_dict[person].append(image_file_path)

        for audio_file_path in audio_files:
            person = audio_file_path.split("/")[-2]
            audio_files_dict[person].append(audio_file_path)

        # print_error(audio_files_dict.keys())
        # print_error(image_files_dict.keys())
        # print_debug(image_files)

        # print_error(self._images_dir.split("/"))

        # exit(-1)

        all_files = []
        for person in image_files_dict.keys():

            audio_files_current_person = audio_files_dict[person]
            image_files_current_person = image_files_dict[person]

            if len(audio_files_current_person) != len(image_files_current_person):
                print("Data for {} does not have matching records!".format(person))
                exit(-1)

            for i in range(len(audio_files_current_person)):
                res = {"label": person, "audio": audio_files_current_person[i], "image": image_files_current_person[i]}
                all_files.append(res)

        self._train_files, self._val_files, _, _ = train_test_split(all_files, all_files, test_size=0.1, random_state=42)

    def get_train_files(self):
        shuffle(self._train_files)
        return self._train_files

    def get_val_files(self):
        shuffle(self._val_files)
        return self._val_files

    def get_test_files(self):
        if self.is_live:
            test_dir = "/tmp/test_asariri/"
            record_audio(test_dir)

            for each in os.listdir(test_dir):
                res = {"label": "test", "audio":  test_dir + "/" + each, "image": "None"}
                self._test_files.append(res)

            print_error(self._test_files)
            return self._test_files
        else:
            shuffle(self._val_files)
            return self._val_files[:10]


    def predict_on_test_files(self, data_iterator, estimator):

        predictions_fn = estimator.predict(input_fn=data_iterator.get_test_input_function(),
                                           hooks=[])

        predictions = []

        for r in tqdm(predictions_fn, desc = "predictions: "):
            images = r
            predictions.append(images)
            my_i = images.squeeze()
            plt.imshow(my_i, cmap="gray_r")
            plt.pause(1)

        input("press enter to exit!")


