from dataset import ImagesDataset
from classifier import PCAClassifier
from skimage import color
import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt

test_images_path = 'C:\\ws\\faces_sets\\faces_sets\\test_set'
test_data = ImagesDataset(path=test_images_path)
test_data.load(verbose=False)

def normalize_gray_pixels(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 100

test_gray_images = [normalize_gray_pixels(color.rgb2gray(img).flatten()) for img in test_data.images]
gray_img = test_gray_images[0]

gray_img = img_as_float(gray_img)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.arange(0, 180), np.arange(0, 200))
ax.plot_surface(x, y, gray_img.reshape(200, 180), cmap='gray')

# turn off the axis planes
ax.set_axis_off()

plt.show()