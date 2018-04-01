import pickle
import os
import tensorflow as tf
from sarvam.helpers.print_helper import *

EXPERIMENT_ROOT_DIR = "experiments"
EXPERIMENT_DATA_ROOT_DIR = EXPERIMENT_ROOT_DIR + "/asariri/data/"
EXPERIMENT_MODEL_ROOT_DIR = EXPERIMENT_ROOT_DIR + "/asariri/models/"

class ModelConfigBase():
    @staticmethod
    def dump(model_dir, config):
        tf.logging.info(CGREEN2 + str("Use this model directory for further retraining "
                   "or prediction (--model-dir) :  " + model_dir) + CEND)
        print_info("Use this model directory for further retraining "
                   "or prediction (--model-dir) :  " + model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(model_dir+"/model_config.pickle", "wb") as file:
            pickle.dump(config, file)

    @staticmethod
    def load(model_dir):
        with open(model_dir + "/model_config.pickle", "rb") as file:
            cfg = pickle.load(file)
        return cfg
