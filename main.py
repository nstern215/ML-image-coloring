# %% import

from dataset import ImagesDataset
from classifier import PCAClassifier
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

# %% load data

train_images_path = 'C:\\ws\\faces_sets\\faces_sets\\training_set'
train_data = ImagesDataset(path=train_images_path)
train_data.load(verbose=True)

# %% normalize gray pixels values to [0, 100]

def normalize_gray_pixels(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 100

# %% preprocess images for PCA

gray_images = [normalize_gray_pixels(color.rgb2gray(img).flatten()) for img in train_data.images]

# gray_images = [color.rgb2lab(img)[:,:,0].flatten() for img in train_data.images]

# %% classifier

clf = PCAClassifier()
clf.fit(gray_images)


# %% load test images

test_images_path = 'C:\\ws\\faces_sets\\faces_sets\\test_set'
test_data = ImagesDataset(path=test_images_path)
test_data.load(verbose=True)

# %% preprocess test images

test_gray_images = [normalize_gray_pixels(color.rgb2gray(img).flatten()) for img in test_data.images]
# test_gray_images = [color.rgb2lab(img)[:,:,0].flatten() for img in test_data.images]

# %% predict similar image

sim_gray_img = clf.predict([test_gray_images[0]])

# %% find RGB version of similar image

for i, img in enumerate(gray_images):
    if np.array_equal(img, sim_gray_img):
        sim_img = train_data.images[i]
        break

# %% plot

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(test_data.images[0])
ax[0, 0].set_title('Test image RGB')
ax[0, 1].imshow(test_gray_images[0].reshape(200, 180), cmap='gray')
ax[0, 1].set_title('Test image grayscale')

ax[1, 0].imshow(sim_img)
ax[1, 0].set_title('Similar image RGB')
ax[1, 1].imshow(sim_gray_img.reshape(200, 180), cmap='gray')
ax[1, 1].set_title('Similar image grayscale')

plt.show()

gray_img = test_gray_images[0]


# %% segmentation

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

gray_img = img_as_float(gray_img)
sim_img = img_as_float(sim_img)
sim_gray_img = img_as_float(sim_gray_img)

gray_image_segments = slic(gray_img.reshape(200, 180), n_segments=5000, compactness=0.1, sigma=1)
sim_gray_img_segments = slic(sim_gray_img.reshape(200, 180), n_segments=5000, compactness=0.1, sigma=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))

# ax[0].imshow(gray_img.reshape(200, 180), cmap='gray')
# ax[0].set_title('Original image')

ax[0].imshow(mark_boundaries(gray_img.reshape(200, 180), gray_image_segments))
ax[0].set_title('Segmented image')

ax[1].imshow(mark_boundaries(sim_img.reshape(200, 180, 3), sim_gray_img_segments))
ax[1].set_title('Segmented similar image')

# plt.show()

# %% analyze segments - features extraction from segments

from scipy.stats import norm

def calc_segment_distribution(gray_img, segment_indexes):

    segment_values = [gray_img[index] for index in segment_indexes]


    counts, bins = np.histogram(segment_values, bins=30)

    mode = bins[np.argmax(counts)]
    median = np.median(segment_values)

    mean, variance = norm.fit(segment_values)
    
    return mean, variance, mode, median


def calc_segment_boundry_rectangle(segment_indexes):
    
    # find boundry rectangle of segment
    min_x = min(segment_indexes, key=lambda x: x[0])[0]
    max_x = max(segment_indexes, key=lambda x: x[0])[0]
    min_y = min(segment_indexes, key=lambda x: x[1])[1]
    max_y = max(segment_indexes, key=lambda x: x[1])[1]

    return min_x, min_y, max_x, max_y

def calc_segment_boundry_rectangle_coverage(segment_indexes, boundry_rectangle):
    
    boundry_rectangle_area = (boundry_rectangle[2] - boundry_rectangle[0]) * (boundry_rectangle[3] - boundry_rectangle[1])
    segment_area = len(segment_indexes)

    try:
        area = segment_area / boundry_rectangle_area
    except ZeroDivisionError:
        area = 0
    return area
    

def find_segment_center(indexes):
    min_x = min(indexes, key=lambda x: x[0])[0]
    max_x = max(indexes, key=lambda x: x[0])[0]
    min_y = min(indexes, key=lambda x: x[1])[1]
    max_y = max(indexes, key=lambda x: x[1])[1]

    return (min_x + max_x) / 2, (min_y + max_y) / 2

def analyze_segment(gray_img, segments):

    segments_map = {}

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            if segments[i][j] not in segments_map:
                segments_map[segments[i][j]] = []

            segments_map[segments[i][j]].append((i,j))

    segments_features = []

    for key in segments_map.keys():
        mean, variance, mode, median = calc_segment_distribution(gray_img, segments_map[key])
        num_of_pixels = len(segments_map[key])

        boundry_rectangle = calc_segment_boundry_rectangle(segments_map[key])
        boundry_rectangle_coverage = calc_segment_boundry_rectangle_coverage(segments_map[key], boundry_rectangle)

        segment_center_x, segment_center_y = find_segment_center(segments_map[key])

        segments_features.append([mean, variance, mode, median, num_of_pixels, boundry_rectangle, boundry_rectangle_coverage, segment_center_x, segment_center_y])

    return segments_map, segments_features

def create_features_matrix(segments_features):

    features_matrix = []

    for segment in segments_features:
        # features = [segment[0], segment[1], segment[2], segment[3], segment[4], segment[6], segment[7], segment[8]]
        # features = [segment[0], segment[1]]
        # features = [segment[0], segment[1], segment[4], segment[6]]

        # features = [segment[0], segment[1], segment[5][0], segment[5][1], segment[7], segment[8]]
        features = [segment[0], segment[1], segment[7], segment[8]]

        features_matrix.append(features)

    return features_matrix



# %% prepare data for classifier

gray_img = gray_img.reshape(200, 180)
sim_gray_img = sim_gray_img.reshape(200, 180)

gray_image_segments_map, gray_image_segments_features = analyze_segment(gray_img, gray_image_segments)
gray_image_features_matrix = create_features_matrix(gray_image_segments_features)

sim_gray_img_segments_map, sim_gray_img_segments_features = analyze_segment(sim_gray_img, sim_gray_img_segments)
sim_ray_image_features_matrix = create_features_matrix(sim_gray_img_segments_features)

sim_img_lab = color.rgb2lab(sim_img)




# %% try find matching segments between images

def find_match_segment(gray_segment_features, sim_gray_features_matrix):

    min_distance = float('inf')
    min_index = -1

    for i, segment_features in enumerate(sim_gray_features_matrix):

        distance = np.linalg.norm(np.array(gray_segment_features) - np.array(segment_features))

        if distance < min_distance:
            min_distance = distance
            min_index = i

    return min_index, min_distance

def map_segment_matching(gray_img_features_matrix, sim_gray_features_matrix):
    
    segments_mapping = {}

    for i, segment_features in enumerate(gray_img_features_matrix):
        min_index, min_distance = find_match_segment(segment_features, sim_gray_features_matrix)

        segments_mapping[i+1] = [min_index, min_distance]

    return segments_mapping

segments_mapping = map_segment_matching(gray_image_features_matrix, sim_ray_image_features_matrix)



# %% drawing segments

# def mark_segment(indexes, image = None):

#     # min_x = min(indexes, key=lambda x: x[0])[0]
#     # min_y = min(indexes, key=lambda x: x[1])[1]

#     # reduce min_x and min_y from indexes
#     # indexes = [(x[0] - min_x, x[1] - min_y) for x in indexes]

#     # max_x = max(indexes, key=lambda x: x[0])[0]
#     # max_y = max(indexes, key=lambda x: x[1])[1]

#     if image is None:
#         pixels = np.zeros((200, 180))

#         for index in indexes:
#             pixels[index[0]][index[1]] = 1

#     else:
#         pixels = image.copy().reshape(200, 180)

#         # all pixels outside the segment are set to 0
#         for i in range(200):
#             for j in range(180):
#                 if (i,j) not in indexes:
#                     pixels[i][j] = 0
    

#     return pixels

# # for each pair of matching segments plot the pair segments of gray_img and sim_img

# for key in segments_mapping.keys():

#     try:
#         gray_img_segment = mark_segment(gray_image_segments_map[key], gray_img)
#         sim_img_segment = mark_segment(sim_gray_img_segments_map[segments_mapping[key][0]], sim_gray_img)

#         plt.subplot(1,2,1)
#         plt.imshow(gray_img_segment, cmap='gray')
#         plt.subplot(1,2,2)
#         plt.imshow(sim_img_segment, cmap='gray')
#         plt.show()
#     except:
#         pass

# %% distances

# plot histogram of distances and show mean and variance in the plot

distances = [segments_mapping[key][1] for key in segments_mapping.keys()]

plt.hist(distances, bins=30)
plt.show()

print('mean: ', np.mean(distances))
print('variance: ', np.var(distances))


# %% create image from segments

def find_best_color_in_segment(segment_indexes, sim_img_lab):
    
    # in the segment find the mode value for a and b channels
    a_values = [sim_img_lab[index][1] for index in segment_indexes]
    b_values = [sim_img_lab[index][2] for index in segment_indexes]

    a_counts, a_bins = np.histogram(a_values, bins=30)
    b_counts, b_bins = np.histogram(b_values, bins=30)

    a_mode = a_bins[np.argmax(a_counts)]
    b_mode = b_bins[np.argmax(b_counts)]

    return a_mode, b_mode


target_img_lab = np.zeros((200, 180, 3))
target_img_lab[:, :, 0] = gray_img.reshape(200, 180)

for key in segments_mapping.keys():
    target_img_lab_segment_indexes = gray_image_segments_map[key]
    
    try:
        sim_img_lab_segment_indexes = sim_gray_img_segments_map[segments_mapping[key][0]]
    
        a_mode, b_mode = find_best_color_in_segment(sim_img_lab_segment_indexes, sim_img_lab)

        for index in target_img_lab_segment_indexes:
            target_img_lab[index][1] = a_mode
            target_img_lab[index][2] = b_mode
    except:
        print('segment not found {}'.format(key))

# %% test segment matching







# %% show results

target_img = color.lab2rgb(target_img_lab)

# plot sim img in the left
# plot gray img in the middle
# plot target img in the right

fig, ax = plt.subplots(1, 3, figsize=(15, 15))

ax[0].imshow(sim_img)
ax[0].set_title('sim img')

ax[1].imshow(gray_img, cmap='gray')
ax[1].set_title('gray img')

ax[2].imshow(target_img)
ax[2].set_title('target img')

plt.show()

# %% calculate error

print('error: ', np.linalg.norm(target_img.flatten() - sim_img.flatten()))

# %% 


from matplotlib.colors import LinearSegmentedColormap

a_cmap = LinearSegmentedColormap.from_list('a_cmap', ['green', 'red'])
b_cmap = LinearSegmentedColormap.from_list('b_cmap', ['blue', 'yellow'])

# %%

# plot sim_img
# than, convert sim_img to lab and plot L channel in gray scale

# plot a channel in red and green
# plot b channel in blue and yellow

fig, ax = plt.subplots(1, 3, figsize=(15, 15))

ax[0].imshow(sim_img)
ax[0].set_title('sim img')

ax[1].imshow(sim_img_lab[:, :, 0], cmap='gray')
ax[1].set_title('L channel')

ax[2].imshow(sim_img_lab[:, :, 1], cmap=a_cmap)
ax[2].set_title('a channel')

plt.show()
