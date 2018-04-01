import os
import sys
import tensorflow as tf
import tarfile
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

from sarvam.helpers.print_helper import *

def maybe_download_and_extract_dataset(data_url, dest_directory):
    """Download and extract data set tar file.
  
    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.
  
    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
        return
    print_info("Checking destination directory : " + dest_directory)
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    else:
        print_info("SR dataset already exists!")
        return
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        try:
            filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        except:
            tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                             filepath)
            tf.logging.error('Please make sure you have enough free space and'
                             ' an internet connection')
            raise
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                        statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)