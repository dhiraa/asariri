import sys
sys.path.append("../")

from importlib import import_module
import asariri.dataset
class DatasetFactory():

    dataset_path = {
        "crawled_dataset": "asariri.dataset.crawled_dataset",
        "mnist_dataset": "asariri.dataset.mnist_dataset",
    }

    datasets = {
        "crawled_dataset": "CrawledData",
        "mnist_dataset": "Mnist",
    }

    def __init__(self):
        pass

    @staticmethod
    def _get_dataset(name):
        try:
            dataset = getattr(import_module(DatasetFactory.dataset_path[name]), DatasetFactory.datasets[name])
        except KeyError:
            raise NotImplemented("Given dataset file name not found: {}".format(name))
        # Return the model class
        return dataset

    @staticmethod
    def get(dataset_name):
        dataset = DatasetFactory._get_dataset(dataset_name)
        return dataset


