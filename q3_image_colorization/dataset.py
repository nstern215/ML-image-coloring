"""
@ Author: Netanel Stern
@ ID: 206342255
@ Date: 12/02/2023
"""

"""
Dataset module for image colorizer

The module loads the data from given path and create a dataset
for the model.
"""



from skimage import io
import numpy as np
import os


class ImagesDataset:

    def __init__(self, path=None, formats=['jpg'], images=None):
        """
        Initialize the dataset

        Parameters
        ----------
        path : str, optional
            The path to the data.
        formats : list, optional
            The formats of the images. The default is ['jpg'].
        images : list, optional

        Returns
        -------
        None.

        """

        self.path = path
        self.formats = formats

        if images is not None:
            self.images = images
        else:
            self.images = []

    def set_path(self, path):
        """
        Set the path of the dataset

        Parameters
        ----------
        path : str
            The path to the data.

        Returns
        -------
        None.

        """

        self.path = path

    def set_formats(self, formats):
        """
        Set the formats of the images

        Parameters
        ----------
        formats : list
            The formats of the images.

        Returns
        -------
        None.

        """

        self.formats = formats

    def load(self, verbose=False):
        """
        Load data for the dataset from given path

        Returns
        -------
        None.

        """

        for fname in os.walk(self.path).__next__()[2]:
            if fname.split('.')[-1] in self.formats:

                img_path = os.path.join(self.path, fname)

                if verbose:
                    print('Loading image: {}'.format(img_path))

                self.images.append(io.imread(img_path))

                if verbose:
                    print('Num of loaded images: {}'.format(len(self.images)))
