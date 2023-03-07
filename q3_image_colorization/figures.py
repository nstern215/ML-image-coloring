"""
@ Author:  Netanel Stern
@ ID:      206342255
"""

"""
This file contains the code that used to generate the figures for the report.
"""
# %% imports

import numpy as np
import matplotlib.pyplot as plt
import image_util as iu
from skimage import color
from skimage.segmentation import slic, mark_boundaries
from dataset import ImagesDataset
from sim_image_colorizer import Colorizer
import os as os

# %% functions

def check_paint_quality(source, target):
    dist = 0

    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            dist += np.linalg.norm(source[i, j] - target[i, j])

    return dist / (source.shape[0] * source.shape[1])

# %% data loading

data_path = 'faces_sets'

# train_images_path = 'C:\\ws\\faces_sets\\faces_sets\\training_set'
train_images_path = os.path.join(data_path, 'training_set')
train_data = ImagesDataset(path=train_images_path)
train_data.load(verbose=False)

colorizer = Colorizer(train_data)
colorizer.fit_data(gray_method='luminosity', verbose=True)

# test_images_path = 'C:\\ws\\faces_sets\\faces_sets\\test_set'
test_images_path = os.path.join(data_path, 'test_set')
test_data = ImagesDataset(path=test_images_path)
test_data.load(verbose=False)

colorizer = Colorizer(train_data)
colorizer.fit_data(gray_method='luminosity')

gray_test_images = [iu.convert_rgb_to_grayscale(img) for img in test_data.images]

# %% figure 3

gray_img = iu.normalize_image(gray_test_images[0], top_val=100, convert_to_int=True)
sim_gray_img = colorizer.clf.predict(gray_img.reshape(1, -1))

for i, img in enumerate(colorizer.gray_images):
    if np.array_equal(img.flatten(), sim_gray_img.flatten()):
        sim_img = colorizer.dataset.images[i]
        break

fig, ax = plt.subplots(1, 3, figsize=(20, 10))

gray_img_segments = colorizer.calc_image_segmentation(gray_img, segments=300, compactness=0.1)
sim_gray_img_segments = colorizer.calc_image_segmentation(sim_gray_img.reshape(200, 180), segments=300, compactness=0.1)

gray_img_lab = np.zeros((200, 180, 3))
gray_img_lab[:, :, 0] = gray_img

ax[0].imshow(gray_img, cmap='gray')
ax[0].axis('off')

ax[1].imshow(mark_boundaries(color.lab2rgb(gray_img_lab), gray_img_segments), cmap='gray')
ax[1].axis('off')

ax[2].imshow(mark_boundaries(sim_img, sim_gray_img_segments))
ax[2].axis('off')

plt.show()

# %% figure 4

sim_img_segments = colorizer.calc_image_segmentation(sim_img, segments=300, compactness=0.1)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(mark_boundaries(color.lab2rgb(gray_img_lab), gray_img_segments), cmap='gray')
ax[0].axis('off')

ax[1].imshow(mark_boundaries(sim_img, sim_img_segments))
ax[1].axis('off')

plt.show()

# %% figure 5
# !!! this code takes a long time to run (can takes up to 2 hours) !!!

segments = [10, 25, 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000]
Y = []

for seg in segments:

    paint_grade = []

    for i, img in enumerate(gray_test_images):
        sim_img, sim_gray_img, target_img = colorizer.colorize(img, verbose=False, target_output_color_space='lab', segments=seg)
        paint_grade.append(check_paint_quality(iu.convert_rgb_to_lab(test_data.images[i]), target_img))

    Y.append(np.mean(paint_grade))

plt.plot(segments, Y)
plt.xlabel('segments #')
plt.ylabel('paint quality')

# %% figure 6

gray_img_100_segments = colorizer.calc_image_segmentation(gray_img, segments=100, compactness=0.1)
gray_img_500_segments = colorizer.calc_image_segmentation(gray_img, segments=500, compactness=0.1)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(mark_boundaries(color.lab2rgb(gray_img_lab), gray_img_100_segments), cmap='gray')
ax[0].axis('off')

ax[1].imshow(mark_boundaries(color.lab2rgb(gray_img_lab), gray_img_500_segments), cmap='gray')
ax[1].axis('off')

plt.show()

# %% figure 7 figure 8

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.arange(0, 180), np.arange(0, 200))
ax.plot_surface(x, y, gray_img.reshape(200, 180), cmap='gray')

plt.show()

# %% figure 11

def remove_person_from_training_set(train_set, person_index):
    new_data = np.delete(train_set.images.copy(), np.arange(person_index * 19, person_index * 19 + 19), axis=0)
    return ImagesDataset(images=new_data)



paint_grade = []
results = {}

for i, img in enumerate(gray_test_images):
    train_set = remove_person_from_training_set(train_data, i)

    colorizer = Colorizer(train_set)
    colorizer.fit_data(gray_method='luminosity', verbose=False)    
    colorized_img = colorizer.colorize(img, verbose=False, target_output_color_space='lab')
    paint_grade.append(check_paint_quality(iu.convert_rgb_to_lab(test_data.images[i]), colorized_img[2]))
        
    colorized_img = (colorized_img[0], colorized_img[1], iu.convert_lab_to_rgb(colorized_img[2]))
    results[i] = colorized_img 

fig, ax = plt.subplots(len(results), 3, figsize=(9, 20))

for i, img in enumerate(results):
        ax[i, 0].imshow(results[img][0])
        ax[i, 0].axis('off')
            
        ax[i, 1].imshow(gray_test_images[i], cmap='gray')
        ax[i, 1].axis('off')
            
        ax[i, 2].imshow(results[img][2])
        ax[i, 2].axis('off')
        if i == 0:
            ax[i, 0].set_title('sim image')
            ax[i, 1].set_title('gray image')
            ax[i, 2].set_title('target image')

plt.show()