import os
import sys

from sarvam.helpers.print_helper import *
from speech_recognition.commands.data_iterator_factory import DataIteratorFactory
from speech_recognition.commands.dataset_factory import DatasetFactory
from speech_recognition.commands.model_factory import ModelsFactory

sys.path.append("../../")
sys.path.append("../")
sys.path.append(".")

list_of_data_iterators_names = DataIteratorFactory.iterators.keys()
list_of_models_names = ModelsFactory.models.keys()

print_info("\n Bundled datasets: (--dataset-name)")
print_info(list(DatasetFactory.datasets.keys()))

print_info("\n Possible Tensorflow Models for each data iterator:")
for data_iterator_name in list_of_data_iterators_names:
    print_info( "\n" + data_iterator_name + "(--data-iterator-name)")
    for tf_model_name in list_of_models_names:
        data_iterator_instance = DataIteratorFactory.get(data_iterator_name)
        data_iterator_instance = data_iterator_instance(None, None, None,None)
        model_cfg, model_instance = ModelsFactory.get(tf_model_name)
        if (data_iterator_instance.feature_type == model_instance.feature_type):
            print_info("\t |--->  "+ tf_model_name + "(--model-name)")