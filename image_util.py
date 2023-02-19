"""
@ Author:  Netanel Stern
@ ID:      206342255
@ Date:    12/02/2023
"""

"""
Image Util provide some useful functions for image processing
"""

from skimage import io, color
import numpy as np

def read_image(path):
    """
    Read an image from the given path

    Parameters
    ----------
    path : str
        The path to the image.

    Returns
    -------
    numpy array
        The image as numpy array.

    """
    return io.imread(path)

def convert_rgb_to_lab(image):
    """
    Convert the image to LAB

    Parameters
    ----------
    image : numpy array
        The image to convert.

    Returns
    -------
    numpy array
        The LAB image.

    """
    return color.rgb2lab(image)

def convert_lab_to_rgb(image):
    """
    Convert the image to RGB

    Parameters
    ----------
    image : numpy array
        The image to convert.

    Returns
    -------
    numpy array
        The RGB image.

    """
    return color.lab2rgb(image)

def convert_grayscale_to_rgb(image):
    """
    Convert the image to RGB

    Parameters
    ----------
    image : numpy array
        The image to convert.

    Returns
    -------
    numpy array
        The RGB image.

    """
    return color.gray2rgb(image)


def convert_rgb_to_grayscale(image, method='average'):
    """
    Convert the image to grayscale
    
    Parameters
    ----------
    image : numpy array
        The image to convert.
    method : str, optional
        The method to use for conversion. The default is 'average'.
        The available methods are:
            'average' - average the three channels.
            'luminosity' - use the luminosity method.
            'red' - use the red channel.
            'green' - use the green channel.
            'blue' - use the blue channel.

    Returns
    -------
    numpy array
        The grayscale image.

    Raises
    ------
    ValueError
        If the method is invalid.
    """

    if method == 'average':
        return color.rgb2gray(image)
    elif method == 'luminosity':
        return color.rgb2lab(image)[:, :, 0]
    elif method == 'red':
        return image[:, :, 0]
    elif method == 'green':
        return image[:, :, 1]
    elif method == 'blue':
        return image[:, :, 2]
    else:
        raise ValueError('Invalid method')

def normalize_image(image, top_val=255, convert_to_int=False):
    """
    Normalize image pixels value to range [0, top_val]
    The return image is in the same shape as the input image

    Parameters
    ----------
    image : numpy array
        The image to normalize.
    top_val : int, optional
        The top value to normalize to. The default is 255.
    convert_to_int : bool, optional
        If True, the image will be converted to int. The default is False.

    Returns
    -------
    numpy array
        The normalized image.
    """

    image_dim = image.shape
    image = image.flatten()

    min_val = np.min(image)
    max_val = np.max(image)

    image = (image - min_val) / (max_val - min_val) * top_val

    if convert_to_int:
        image = image.astype(int)

    return image.reshape(image_dim)

def calc_avergae_image(images):
    """
    Calculate the average image from the given images

    Parameters
    ----------
    images : list of numpy array
        The images to calculate the average from.

    Returns
    -------
    numpy array
        The average image.

    """
    
    return np.mean(images, axis=0)
