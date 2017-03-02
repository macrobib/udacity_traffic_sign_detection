import cv2
import numpy as np
import csv
import pickle
import random


class augment:
    """Class to do augmentation of images."""
    def __init__(images, convert=False):
        self.images = images # The dataset.
        self.convert = convert # convert to grayscale.


    def translation(image, trans_range=3):
        """Image translation"""
        row, col, _ = image.shape
        tr_x = trans_range*np.random.uniform() - trans_range/2
        tr_y = trans_range*np.random.uniform() - trans_range/2

        M = np.float32([1, 0 , tr_x], [0, 1, tr_y])
        return cv2.warpAffine(image, M, (col, row))

    def rotation(image, val=10):
        """Rotate the image."""
        row, col, _ = image.shape



