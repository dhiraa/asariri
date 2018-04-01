import sys
sys.path.append("../")

import tensorflow as tf
import argparse
from importlib import import_module

class DataIteratorFactory():

    iterator_path = {
        "crawled_data_iterator": "asariri.dataset.iterators.crawled_data_iterator",
        "mnist_iterator" : "asariri.dataset.iterators.mnist_iterator"
    }

    iterators = {
        "crawled_data_iterator": "CrawledDataIterator",
        "mnist_iterator": "MnistDataIterator"
    }


    def __init__(self):
        pass

    @staticmethod
    def _get_iterator(name):
        '''
        '''
        try:
            data_iterator = getattr(import_module(DataIteratorFactory.iterator_path[name]), DataIteratorFactory.iterators[name])
        except KeyError:
            raise NotImplemented("Given data iterator file name not found: {}".format(name))
        # Return the model class
        return data_iterator

    @staticmethod
    def get(iterator_name):
        iterator = DataIteratorFactory._get_iterator(iterator_name)
        return iterator


