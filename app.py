# %% main

from sim_image_colorizer import Colorizer
from dataset import ImagesDataset
import image_util as iu
import matplotlib.pyplot as plt
import numpy as np
import os as os

def check_paint_quality(source, target):
    dist = 0

    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            dist += np.linalg.norm(source[i, j] - target[i, j])

    return dist / (source.shape[0] * source.shape[1])

train_images_path = 'C:\\ws\\faces_sets\\faces_sets\\training_set'
train_data = ImagesDataset(path=train_images_path)
train_data.load(verbose=False)

test_images_path = 'C:\\ws\\faces_sets\\faces_sets\\test_set'
test_data = ImagesDataset(path=test_images_path)
test_data.load(verbose=False)

colorizer = Colorizer(train_data)
colorizer.fit_data(gray_method='luminosity')

gray_test_images = [iu.convert_rgb_to_grayscale(img) for img in test_data.images]
avg_img = iu.calc_avergae_image(train_data.images)

paint_grade_sim = []
paint_grade_avg = []

for i, img in enumerate(gray_test_images):
    sim_img, sim_gray_img, target_img = colorizer.colorize(img, target_output_color_space='lab')
    paint_grade_sim.append(check_paint_quality(iu.convert_rgb_to_lab(test_data.images[i]), target_img))

    target_img = iu.convert_lab_to_rgb(target_img)

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].imshow(sim_gray_img, cmap='gray')
    ax[0, 0].set_title('Similar image in gray')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(img, cmap='gray')
    ax[0, 1].set_title('target image in gray')
    ax[0, 1].axis('off')

    ax[1, 0].imshow(sim_img)
    ax[1, 0].set_title('Similar image')
    ax[1, 0].axis('off')

    ax[1, 1].imshow(target_img)
    ax[1, 1].set_title('target image')
    ax[1, 1].axis('off')

    plt.show()

for i, img in enumerate(gray_test_images):
    avg_img, avg_gray_img, target_img = colorizer.colorize(img, target_output_color_space='lab', ref_img=avg_img, verbose=False)
    paint_grade_avg.append(check_paint_quality(iu.convert_rgb_to_lab(test_data.images[i]), target_img))

    target_img = iu.convert_lab_to_rgb(target_img)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('target image in gray')
    ax[0].axis('off')

    ax[1].imshow(avg_img)
    ax[1].set_title('Average image')
    ax[1].axis('off')

    ax[2].imshow(target_img)
    ax[2].set_title('target image')
    ax[2].axis('off')

    plt.show()

# %% figure 10
ax = plt.subplot(111)

# diff_sim_img = [15.364884885232081,10.361834027300835,12.428067830165814,12.114834200712716,16.126676657932453,10.38086614618144,13.39572247299863]
diff_sim_img = [16.11990538481141,
 14.373619867197606,
 14.07947755912508,
 15.306607495624698,
 16.428777260351985,
 11.234692616817878,
 14.750202801354135]

X = np.arange(len(paint_grade_sim))

ax.bar(X, paint_grade_sim, width=0.2, label='Similar image')
ax.bar(X + 0.2, paint_grade_avg, width=0.2, label='Average image')
ax.bar(X + 0.4, diff_sim_img, width=0.2, label='Diff sim image')

# ax.legend(loc='upper center')
ax.legend(loc='best')
plt.show()