
# %% imports

from sim_image_colorizer import Colorizer
from dataset import ImagesDataset
import image_util as iu
import matplotlib.pyplot as plt
import numpy as np
import os as os

# if __name__ == '__main__':

def check_paint_quality(source, target):
    dist = 0

    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            dist += np.linalg.norm(source[i, j] - target[i, j])

    return dist / (source.shape[0] * source.shape[1])

train_images_path = 'C:\\ws\\faces_sets\\faces_sets\\training_set'
train_data = ImagesDataset(path=train_images_path)
train_data.load(verbose=True)

colorizer = Colorizer(train_data)
colorizer.fit_data(gray_method='luminosity', verbose=True)

test_images_path = 'C:\\ws\\faces_sets\\faces_sets\\test_set'
test_data = ImagesDataset(path=test_images_path)
test_data.load(verbose=True)




gray_test_images = [iu.convert_rgb_to_grayscale(img) for img in test_data.images]

# %
# img = gray_test_images[0]

# sim_img, sim_gray_img, target_img = colorizer.colorize(img, verbose=True)

# for each image create a new figure and plot in one row the sim_img, img in gray and target_img
# fig, ax = plt.subplots(1, 3, figsize=(20, 10))

# ax[0].imshow(sim_img)
# ax[0].set_title('Similar image')

# ax[1].imshow(img, cmap='gray')
# ax[1].set_title('target image in gray')

# ax[2].imshow(target_img)
# ax[2].set_title('target image')

# plt.show()

# %% calc avg

avg = iu.calc_avergae_image(test_data.images)

# normalize the avg image
# avg = iu.normalize_image(avg, top_val=255, convert_to_int=True)
avg = avg.astype(int)

# plot the avg image
plt.imshow(avg)
# axis off
plt.axis('off')

plt.show()

# %%

# sim_img, sim_gray_img, target_img = colorizer.colorize(gray_test_images[0], ref_img=avg, verbose=True, target_output_color_space='rgb', segments=350, compactness=0.5, sigma=1)

# target_img = iu.normalize_image(target_img, top_val=255, convert_to_int=True)

# plt.imshow(target_img)


# %%

from skimage import color



# segments = kwargs.pop('segments', 250)            
# compactness = kwargs.pop('compactness', 0.1)
# sigma = kwargs.pop('sigma', 1)

args = {
    "segments": 300,
    "compactness": 0.1,
    "sigma": 1
}

# segments = [10, 25, 50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
# segments = [1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
# segments = [2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000]

segments = [350]

Y = []
for seg in segments:
    print('segments: {}'.format(seg))

    # check if the folder exists
    # if not os.path.exists('{}'.format(seg)):
    #     os.mkdir('{}'.format(seg))

    paint_grade = []
    for i, img in enumerate(gray_test_images):

    # img = gray_test_images[0]
        sim_img, sim_gray_img, target_img = colorizer.colorize(img, ref_img=avg, verbose=False, target_output_color_space='rgb', segments=seg, compactness=0.5, sigma=1)

        target_img = iu.normalize_image(target_img, top_val=255, convert_to_int=True)

        # paint_grade.append(check_paint_quality(iu.convert_rgb_to_lab(test_data.images[i]), target_img))

            # save the target image array content as text file
            # np.savetxt('{}/target_img_{}.txt'.format(seg, i), target_img.flatten())


            # check if img and sim_gray_img are the same
            # print('Are the images the same?')
            # print(np.array_equal(img, sim_gray_img))

        # fig, ax = plt.subplots(2, 2, figsize=(7, 7))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('target image in gray')
        ax[0].axis('off')

        ax[1].imshow(target_img)
        ax[1].set_title('target image')
        ax[1].axis('off')

        # ax[0, 0].imshow(sim_gray_img, cmap='gray')
        # ax[0, 0].set_title('Similar image in gray')
        # ax[0, 0].axis('off')

        # ax[0, 1].imshow(img, cmap='gray')
        # ax[0, 1].set_title('target image in gray')
        # ax[0, 1].axis('off')

        # ax[1, 0].imshow(sim_img)
        # ax[1, 0].set_title('Similar image')
        # ax[1, 0].axis('off')

        # ax[1, 1].imshow(target_img)
        # ax[1, 1].set_title('target image')
        # ax[1, 1].axis('off')

        plt.show()

            
    


    # print('The average paint grade is: {}'.format(np.mean(paint_grade)))

    # Y.append(np.mean(paint_grade))

# %% results

# plot the results
plt.plot(segments, Y)

# %%

# find the index of the minimum value in Y
min_index = np.argmin(Y)
print('The minimum value is: {}'.format(Y[min_index]))
print('The index of the minimum value is: {}'.format(min_index))

# %% performance

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



# for seg in segments:

#     print('segments: {}'.format(seg))

#     paint_grade = []
#     for i in range(7):
#         target_img = np.loadtxt('{}/target_img_{}.txt'.format(seg, i))
#         target_img = target_img.reshape(200, 180, 3)
#         paint_grade.append(check_paint_quality(iu.convert_rgb_to_lab(test_data.images[i]), target_img))

#         # if i == 0 convert the target_img to rgb plot it
#         if i == 0:
#             fig, ax = plt.subplots(1, 1, figsize=(5,5))

#             ax.imshow(iu.convert_lab_to_rgb(target_img))
#             ax.set_title('target image')
#             ax.axis('off')

#             plt.show()

#     Y.append(np.mean(paint_grade))

# %% results

# plot the results
plt.plot(segments, Y)

# set axis labels
plt.xlabel('segments #')
plt.ylabel('paint quality')
