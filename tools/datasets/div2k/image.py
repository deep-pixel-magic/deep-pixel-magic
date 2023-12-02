import os
from PIL import Image

from tools.datasets.div2k import div2k


class Div2kImage:
    """Represents a single image from the DIV2K dataset."""

    def __init__(self, root: str, dataset_info: div2k.Info):
        """Constructor.

        Args:
            root: The data root directory.
            dataset_info: The dataset info.
        """

        self.root = root
        self.info = dataset_info

    def load(self, image_id: int):
        """Loads an image from the dataset based on the given id.

        Args:
            image_id: The image id.
        """

        path = self.__image_path(self.info, image_id)
        self.image = Image.open(path).convert("RGB")

    def save(self, path: str):
        """Saves the image to the specified path.

        Args:
            path: The path to save the image to.
        """

        self.image.save(path, "PNG")

    def scale(self, scale: int, resample=Image.BICUBIC):
        """Scales the image.

        Args:
            scale: The scale factor.
            resample: The resampling filter to use.
        """

        new_width = self.image.size[0] * 4
        new_height = self.image.size[1] * 4

        new_size = (new_width, new_height)

        self.image = self.image.resize(new_size, resample)

    def get(self):
        """Gets the internal image object.

        Returns:
            The internal image object.
        """

        return self.image

    def __image_path(self, info: div2k.Info, image_id: int):
        """Constructs the image path based on the dataset info and the image id.

        Args:
            info: The dataset info.
            image_id: The image id.

        Returns:
            The path to the image.
        """

        sub_path = ''

        if info.resolution == div2k.Resolution.LOW or info.resolution == 'LR':
            sub_path = f'DIV2K_{info.subset}_{info.resolution}_{info.sampling}/{info.scale}/{image_id:04}{info.scale.lower()}.png'
        else:
            sub_path = f'DIV2K_{info.subset}_{info.resolution}/{image_id:04}.png'

        return os.path.join(self.root, sub_path)
