"""Provides functionality to download the DIV2K dataset."""

from enum import Enum

import os
import urllib.request
import zipfile


class Scale(Enum):
    """Describes all available downsampling scales for the DIV2K dataset."""

    X1 = 'X1'
    """Specifies that the images are not downsampled."""

    X2 = 'X2'
    """Specifies that the images are downsampled by a factor of 2."""

    X3 = 'X3'
    """Specifies that the images are downsampled by a factor of 3."""

    X4 = 'X4'
    """Specifies that the images are downsampled by a factor of 4."""


class Subset(Enum):
    """Describes all available subsets of the DIV2K dataset."""

    TRAINING = 'train'
    """Specifies that the training subset is used."""

    VALIDATION = 'valid'
    """Specifies that the validation subset is used."""

    TESTING = 'test'
    """Specifies that the testing subset is used."""


class Sampling(Enum):
    """Describes all downsampling methods available for the DIV2K dataset."""

    UNKNOWN = 'unknown'
    """Specifies that the downsampling method is unknown."""

    BICUBIC = 'bicubic'
    """Specifies that the bicubic downsampling method is used."""


class Resolution(Enum):
    """Describes all available resolutions of the DIV2K dataset."""

    LOW = 'LR'
    """Specifies that the low resolution images are used."""

    HIGH = 'HR'
    """Specifies that the high resolution images are used."""


class Info:
    """Represents information identifying a specifc dataset of the DIV2K collection."""

    def __init__(self,
                 scale=Scale.X1,
                 subset=Subset.TRAINING,
                 sampling=Sampling.BICUBIC,
                 resolution=Resolution.LOW):
        """Constructor.

        Args:
            scale: The downsampling scale.
            subset: The data subset.
            sampling: The downsampling method.
            resolution: The image resolution.
        """

        self.scale = scale
        self.subset = subset
        self.sampling = sampling
        self.resolution = resolution


class Dataset:
    """Represents a specific instance of the DIV2K dataset."""

    def __init__(self,
                 info=Info(),
                 data_dir=".div2k"):
        """Constructor.

        Args:
            info: A DatasetInfo object describing the dataset to be downloaded.
            data_dir: The directory where the dataset is downloaded to.
        """

        self.info = info
        self.data_dir = data_dir

    def download(self):
        """Downloads the dataset to the data directory."""

        archive_info = self.info
        archive_name = self.__archive_name(
            archive_info.scale, archive_info.subset, archive_info.sampling, archive_info.resolution)

        self.__download_archive(archive_name)

    def __download_archive(self, file):
        """Downloads the specified archive to the data directory.

        Args:
            file: The name of the archive to be downloaded.
        """

        target_dir = os.path.abspath(self.data_dir)
        target_file = os.path.join(target_dir, file)

        source_url = f'http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}'
        self.__log(f'downloading dataset from: {source_url}')

        os.makedirs(target_dir, exist_ok=True)

        def reporter(blocknum, blocksize, totalsize):
            progress = blocknum * blocksize / totalsize * 100
            self.__log(
                f'downloading dataset: {progress:.2f} %', end='\n', flush=True)

        urllib.request.urlretrieve(
            source_url, target_file, reporthook=reporter)

        self.__log('', flush=True)
        self.__log('unzipping dataset ...')

        with zipfile.ZipFile(target_file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        self.__log('removing downloaded archive ...')
        os.remove(target_file)

        self.__log('done')

    def __archive_name(self, scale, subset, sampling, resolution):
        """Uses the specified information to generate the name of the archive to be downloaded.

        Args:
            scale: The downsampling scale.
            subset: The data subset.
            sampling: The downsampling method.
            resolution: The image resolution.
        """

        if resolution == Resolution.HIGH:
            return f'DIV2K_{subset.value}_{resolution.value}.zip'

        if scale == Scale.X1:
            return f'DIV2K_{subset.value}_{resolution.value}_{sampling.value}.zip'

        return f'DIV2K_{subset.value}_{resolution.value}_{sampling.value}_{scale.value}.zip'

    def __log(self, message, end='\n', flush=False):
        """Prints the specified message to the console, using a predefined prefix.

        Args:
            message: The message to be printed.
            end: The string to be appended to the end of the message.
            flush: Specifies whether the output buffer should be flushed after printing the message.
        """

        print("div2k: " + message, end=end, flush=flush)
