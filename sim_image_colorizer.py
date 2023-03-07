"""
@ Author:  Netanel Stern
@ ID:      206342255
@ Date:    12/02/2023
"""

"""
Sim image colorizer module use to colorize the image using image similarity

the similarity is calculated using PCA space with grayscale version of the images
"""

# %% imports

from dataset import ImagesDataset
from classifier import PCAClassifier
import numpy as np
import image_util as iu
from scipy.stats import norm

# %% Sim image colorizer class
class Colorizer:
    """
    Sim image colorizer class use to colorize the image using image similarity

    the similarity is calculated using PCA space with grayscale version of the images

    Attributes
    ----------
    dataset : Dataset
        The dataset to use for colorization.
    """

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset to use for colorization.
        pca_formatter : PCAFormatter
            The PCA formatter to use for colorization.
        """
        self.dataset = dataset
        self.clf = PCAClassifier()

    def fit_data(self, gray_method='average', verbose=False):
        """
        Fit the data to the classifier

        Parameters
        ----------
        gray_method : str, optional
            The method to use for conversion. The default is 'average'.
            The available methods are:
                'average' - average the three channels.
                'luminosity' - luminosity method.
        verbose : bool, optional
            Print the progress. The default is False.
        """

        self.gray_images = [iu.convert_rgb_to_grayscale(img, method=gray_method) for img in self.dataset.images]

        if gray_method == 'luminosity':
            self.gray_images = [iu.normalize_image(img, 100, True) for img in self.gray_images]

        self.clf.fit(self.gray_images)

    def calc_image_segmentation(self, image, method='slic', **kwargs):
        """
        Calculate the image segmentation

        Parameters
        ----------
        image : numpy array
            The image to segment.
        method : str, optional
            The segmentation method to use. The default is 'slic'.
            The available methods are:
                'slic' - SLIC method.
        **kwargs - The arguments for the segmentation method.

        Returns
        -------
        numpy array
            The segmentation of the image.

        Raises
        ------
        Exception
            If the method is unknown.
        
        """

        if method == 'slic':

            from skimage.segmentation import slic

            segments = kwargs.pop('segments', 350)            
            compactness = kwargs.pop('compactness', 0.5)
            sigma = kwargs.pop('sigma', 1)

            print(f'Calculating segmentation with segments={segments}, compactness={compactness}, sigma={sigma}')

            return slic(image, n_segments=segments, compactness=compactness, sigma=sigma)

        else:
            raise Exception(f'Unknown segmentation method: {method}')

    def calc_segment_distribution(self, image, segment_indexes, bins=30):
        """
        Calculate the distribution of the segments

        Parameters
        ----------
        image : numpy array
            The image to segment.
        segment_indexes : list
            The indexes of the segments in the image.
        bins : int, optional
            The number of bins to use. The default is 30.

        Returns
        -------
        list
            The distribution of the segments.

        """

        segment_values = [image[index] for index in segment_indexes]

        counts, bins = np.histogram(segment_values, bins=bins)

        mode = bins[np.argmax(counts)]
        median = np.median(segment_values)

        mean, variance = norm.fit(segment_values)
        
        return mean, variance, mode, median

    def calc_segment_boundry_rectangle(self, segment_indexes):
        """
        Calculate the boundry rectangle of the segment

        Parameters
        ----------
        segment_indexes : list
            The indexes of the segments in the image.

        Returns
        -------
        tuple
            The boundry rectangle of the segment.

        """

        rows = [index[0] for index in segment_indexes]
        cols = [index[1] for index in segment_indexes]

        return min(rows), min(cols), max(rows), max(cols)
    
    def calc_segment_boundry_rectangle_coverage(self, segment_indexes, boundry_rectangle):
        """
        Calculate the boundry rectangle coverage by the segment

        Parameters
        ----------
        segment_indexes : list
            The indexes of the segments in the image.
        boundry_rectangle : tuple
            The boundry rectangle of the segment.

        Returns
        -------
        float
            The boundry rectangle coverage of the segment.

        """

        segment_indexes = set(segment_indexes)

        boundry_rectangle_area = (boundry_rectangle[2] - boundry_rectangle[0]) * (boundry_rectangle[3] - boundry_rectangle[1])
        segment_area = len(segment_indexes)

        try:
            area = segment_area / boundry_rectangle_area
        except ZeroDivisionError:
            area = 0
        return area

    def calc_segment_center(self, segment_indexes):
        """
        Calculate the center of the segment

        Parameters
        ----------
        segment_indexes : list
            The indexes of the segments in the image.

        Returns
        -------
        tuple
            The center of the segment.

        """

        rows = [index[0] for index in segment_indexes]
        cols = [index[1] for index in segment_indexes]

        return (min(rows) + max(rows)) / 2, (min(cols) + max(cols)) / 2

    def calc_segment_map(self, segments):
        """
        Calculate the segment map between the segments and the related pixels

        Parameters
        ----------
        segments : numpy array
            The segments of the image.

        Returns
        -------
        dict
            The segment map.

        """

        segments_map = {}

        for i in range(len(segments)):
            for j in range(len(segments[i])):
                if segments[i][j] not in segments_map:
                    segments_map[segments[i][j]] = []

                segments_map[segments[i][j]].append((i,j))
        
        return segments_map

    def extract_image_features(self, image, segments):
        """
        Extract the features of the image

        Parameters
        ----------
        image : numpy array
            The image to segment.
        segments : numpy array
            The segments of the image.

        Returns
        -------
        list
            The features of the image.

        """

        segments_map = self.calc_segment_map(segments)

        features = []

        for segment in segments_map:
            segment_indexes = segments_map[segment]

            mean, variance, mode, median = self.calc_segment_distribution(image.reshape(200, 180), segment_indexes)
            boundry_rectangle = self.calc_segment_boundry_rectangle(segment_indexes)
            boundry_rectangle_coverage = self.calc_segment_boundry_rectangle_coverage(segment_indexes, boundry_rectangle)
            center = self.calc_segment_center(segment_indexes)

            num_of_pixels = len(segment_indexes)

            features.append([mean, variance, mode, median, num_of_pixels, boundry_rectangle[0], boundry_rectangle[1], boundry_rectangle[2], boundry_rectangle[3], boundry_rectangle_coverage, center[0], center[1]])

        return segments_map, features

    def create_features_matrix(self, features):
        """
        Create the features matrix

        Parameters
        ----------
        features : list
            The features of the image.

        Returns
        -------
        numpy array
            The features matrix.

        """

        features_matrix = []

        for feature in features:
            # features_matrix.append([feature[0], feature[10], feature[11]])
            features_matrix.append([feature[0], feature[1], feature[2], feature[3], feature[4], feature[5], feature[6], feature[7], feature[8], feature[9]])

        return np.array(features_matrix)

    def find_match_segment(self, gray_segment_features, sim_gray_features_matrix):
        """
        Find the matching segments between the gray image and the gray version of the similar image

        Parameters
        ----------
        gray_segment_features : list
            The features of the gray segments.
        sim_gray_features_matrix : numpy array
            The features matrix of the gray segments.

        Returns
        -------
        int
            The index of the matching segment.
        float
            The distance between the segments.

        """

        min_distance = float('inf')
        min_index = -1

        for i, segment_features in enumerate(sim_gray_features_matrix):

            distance = np.linalg.norm(np.array(gray_segment_features) - np.array(segment_features))

            if distance < min_distance:
                min_distance = distance
                min_index = i+1

        return min_index, min_distance

    def map_segment_matching(self, gray_img_features_matrix, sim_gray_features_matrix):
        """
        Map the segments between the gray image and the matching segments in the gray version of the similar image

        Parameters
        ----------
        gray_img_features_matrix : numpy array
            The features matrix of the gray segments of the image.
        sim_gray_features_matrix : numpy array
            The features matrix of the gray segments of the similar image.

        Returns
        -------
        dict
            The mapping between the segments.

        """

        segments_mapping = {}

        for i, segment_features in enumerate(gray_img_features_matrix):
            min_index, min_distance = self.find_match_segment(segment_features, sim_gray_features_matrix)

            segments_mapping[i+1] = [min_index, min_distance]

        return segments_mapping

    def find_color_by_luminosity(self, l_val, sim_image_lab, sim_segment_indexes):
        """
        In the given segment in the similar image, find the closest color to the given luminosity value

        Parameters
        ----------
        l_val : float
            The luminosity value of the pixel to paint.
        sim_image_lab : numpy array
            The LAB version of the similar image.
        sim_segment_indexes : list
            The indexes of the segment in the similar image.

        Returns
        -------
        tuple
            The a and b values of the color.

        """

        a = 0
        b = 0
        
        l_distance = np.inf

        for index in sim_segment_indexes:
            l = sim_image_lab[index][0]
            distance = np.abs(l - l_val)

            if distance < l_distance:
                l_distance = distance
                a = sim_image_lab[index][1]
                b = sim_image_lab[index][2]
        
        return a, b

    def colorize(self, image, ref_img=None, segmentation_method='slic', target_output_color_space='rgb', verbose=True, **kwargs):
        """
        Colorize the image using the dataset

        The methods return:
            1. The similar image - colorized image
            2. The similar image - gray image
            3. The colorized input image

        Parameters
        ----------
        image : numpy array
            The gray image to colorize.
        ref_img : numpy array, optional
            The reference image to use. The default is None -> calc the most similar image.
        segmentation_method : str, optional
            The segmentation method to use. The default is 'slic'.
            The available methods are:
                'slic' - SLIC method.

                # todo: add more methods
        verbose : bool, optional
            Whether to print the progress. The default is True.

        **kwargs - additional arrguments.

        Returns
        -------
        numpy array
            The colorized image.
        numpy array
            The gray version of the colorized image.
        numpy array
            The colorized input image.

        Raises
        ------
        Exception
            If no similar image was found in the dataset.

        """

        if verbose:
            print('kwargs: {}'.format(kwargs))

        # if verbose:
        #     print('Converting image to grayscale...')

        # grayscale_image = iu.convert_rgb_to_grayscale(image, method=gray_method)
        # grayscale_image = iu.normalize_image(grayscale_image, top_val=100, convert_to_int=True)

        grayscale_image = iu.normalize_image(image, top_val=100, convert_to_int=True)

        if verbose:
            print('searching for similar image...')

        if ref_img is not None:
            if verbose:
                print('using given reference image')

            sim_gray_img = iu.convert_rgb_to_grayscale(ref_img, method='luminosity')
            sim_gray_img = iu.normalize_image(sim_gray_img, top_val=100, convert_to_int=True)
            sim_gray_img = sim_gray_img.flatten()
            sim_gray_img = sim_gray_img.astype(np.int32)
        else:
            sim_gray_img = self.clf.predict([grayscale_image.flatten()])

        sim_img = None

        if ref_img is not None:
            sim_img = ref_img.astype(np.uint8)
        else:
            if verbose:
                print('searching for similar image colorful version in dataset...')

            for i, img in enumerate(self.gray_images):
                if np.array_equal(img.flatten(), sim_gray_img.flatten()):
                    sim_img = self.dataset.images[i]
                    break

            if sim_img is None:
                raise Exception('No similar image (color) was found in the dataset')

        if verbose:
            print('segmenting image...')
        gray_image_segments = self.calc_image_segmentation(grayscale_image.reshape(200, 180), method=segmentation_method, **kwargs)
        sim_gray_image_segments = self.calc_image_segmentation(sim_gray_img.reshape(200, 180), method=segmentation_method, **kwargs)

        if verbose:
            print('extracting features from images...')

        gray_image_segments_map, gray_image_features = self.extract_image_features(grayscale_image, gray_image_segments)
        gray_image_features_matrix = self.create_features_matrix(gray_image_features)

        sim_gray_image_segments_map, sim_image_features = self.extract_image_features(sim_gray_img, sim_gray_image_segments)
        sim_image_features_matrix = self.create_features_matrix(sim_image_features)

        sim_img_lab = iu.convert_rgb_to_lab(sim_img)

        if verbose:
            print('maching segments between images...')

        segment_mapping = self.map_segment_matching(gray_image_features_matrix, sim_image_features_matrix)

        if verbose:
            print('colorizing image...')

        target_img_lab = np.zeros((200, 180, 3), dtype=np.float64)
        target_img_lab[:, :, 0] = grayscale_image.reshape(200, 180)

        for key in segment_mapping.keys():
            target_img_lab_segment_indexes = gray_image_segments_map[key]

            for index in target_img_lab_segment_indexes:
                l_val = target_img_lab[index][0]
                a, b = self.find_color_by_luminosity(l_val, sim_img_lab, sim_gray_image_segments_map[segment_mapping[key][0]])
                target_img_lab[index][1] = a
                target_img_lab[index][2] = b


        if target_output_color_space == 'rgb':
            if verbose:
                print('converting final image to RGB...')
            target_img = iu.convert_lab_to_rgb(target_img_lab)
        elif target_output_color_space == 'lab':
            if verbose:
                print('converting final image to LAB...')
            target_img = target_img_lab

        if verbose:
            print('done!')

        return sim_img.reshape(200, 180, 3), sim_gray_img.reshape(200, 180), target_img