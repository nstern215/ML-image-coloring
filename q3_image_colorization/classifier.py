"""
@ Author:  Netanel Stern
@ ID:      206342255
@ Date:    12/02/2023
"""

"""
Classifier module for image colorizer uses to find the most similar image
to the input image.

this classifier uses PCA method to reduce the dimension of the images
and use the euclidean or cosine distance to find the most similar image.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

class PCAClassifier:

    def __init__(self, **kwargs):
        """
        Initialize the classifier
        """
        self.pca = PCA()

    def fit(self, X):
        """
        Fit the classifier to the data

        Parameters
        ----------
        X : numpy array
            The data to fit the classifier.

        Returns
        -------
        None.

        """

        X = np.array(X)

        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        X_pca = self.pca.fit_transform(X)
        self.components_ = self.pca.components_

        self.mapping = [(x, x_pca) for x, x_pca in zip(X, X_pca)]

    def get_mapping(self):
        """
        Get the mapping of images and pca

        Returns
        -------
        list
            The mapping of images and pca.

        """
        return self.mapping


    def predict(self, X):
        """
        Predict the most similar image to the input image

        Parameters
        ----------
        X : numpy array
            The image to predict.

        Returns
        -------
        numpy array
            The most similar image to the input image.

        """
        X_pca = self.pca.transform(X)

        return self._euclidean(X_pca)

    def _euclidean(self, X_pca):
        """
        Predict the most similar image to the input image using euclidean distance

        Parameters
        ----------
        X_pca : numpy array
            The image to predict.

        Returns
        -------
        numpy array
            The most similar image to the input image.

        """
        min_dist = np.inf
        min_img = None

        for img, img_pca in self.mapping:
            dist = np.linalg.norm(img_pca - X_pca)
            if dist < min_dist:
                min_dist = dist
                min_img = img

        return min_img