"""
@ Author:  Netanel Stern
@ ID:      206342255
@ Date:    12/02/2023
"""

"""
Image Util provide some useful functions for image processing
"""

from skimage import io, color

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

def convert_rgb_to_grayscale(image):
    """
    Convert the image to grayscale

    Parameters
    ----------
    image : numpy array
        The image to convert.

    Returns
    -------
    numpy array
        The grayscale image.

    """
    return color.rgb2gray(image)

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