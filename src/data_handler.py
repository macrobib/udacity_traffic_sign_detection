import cv2
import numpy as np
import csv
import pickle
import random
from sklearn import preprocessing
from sklearn import train_test_split
import tqdm
from pathlib import Path

class augment:
    """Class to do augmentation of images."""
    def __init__(self, images, convert=False):
        self.images = images # The dataset.
        self.convert = convert # convert to grayscale.


    def translation(self, image, trans_range=3):
        """Image translation"""
        row, col, _ = image.shape
        tr_x = trans_range*np.random.uniform() - trans_range/2
        tr_y = trans_range*np.random.uniform() - trans_range/2

        M = np.float32([1, 0 , tr_x], [0, 1, tr_y])
        return cv2.warpAffine(image, M, (col, row))

    def rotation(self, image, val=10):
        """Rotate the image."""
        row, col, _ = image.shape
        rorate_by = random.randint(5, 5+val)
        M = cv2.getRotationMatrix2D((col/2, row/2), rotate_by, 1)
        dst = cv2.warpAffine(dst, M, (col, row))
        return dst

    def shear(self, image, shear_val=5):
        row, col, _ = image.shape
        point1 = np.float32([[5, 5], [20, 5], [5, 20]])
        p1 = 5 + shear_val * np.random_uniform() - shear_val/2
        p2 = 20 + shear_val * np.random_uniform() - shear_val/2
        point2 = np.float32([[p1, 5], [p2, p1], [5, p2]])
        M = cv2.getAffineTransform(point1, point2)
        dst = cv2.warpAffine(image, M, (col, row))
        return dst

    def random_shift_color_channels(self, image, val=10):
        enable = False
        dst = image
        if enable:
            R = image[:, :, 0]
            G = image[:, :, 1]
            B = image[:, :, 2]
            dst = dstack(np.roll(R, random.randint(-2, val), axis=0),
                        np.roll(G, random.randint(-2, val), axis=1)
                        np.roll(B, random.randint(-2, val), axis=0))
        return dst

    def flip(self, image):
        """Randomly flip the image."""
        enable = False
        dst = image
        if enable:
            dst = cv2.flip(image, randomint(0,1))
        return dst

    def define_aug_functions(self):
        self.data_aug_functions = []
        data_aug_functions[0] = self.rotation
        data_aug_functions[1] = self.translation
        data_aug_functions[2] = self.shear
        data_aug_functions[3] = self.flip
        data_aug_functions[4] = self.random_shift_color_channels


    def augment_image(self, image):
        """Apply augmentation functions."""
        dst = image
        define_aug_functions()
        val = [val for in range(5)]
        random.shuffle(val)
        for i in val:
            dst = data_augment_functions[i](dst)
        return dst

    def augment_data(self, X_train, y_train):
        list_X_train = X_train.tolist()
        list_y_train = y_train.tolist()
        self.label_count = {}

        data_len = len(X_train)
        assert data_len == len(y_train)

        # Create a count of the labels from signnames.csv
        with open('../data/signnames.csv', 'r') as file:
            csvfile = csv.reader(file)
            for line, row in enumerate(csvfile):
                if line != 0:
                    self.content[int(row[0])] = [row[1], 0]
            for i in y_train:
                content[i][1] += 1

        labelwise_data_count = {i:8000-self.content[i][1] for i in content.keys()}
        augment_count = sum(labelwise_data_count.values())

        while augment_count > 0:
            for i in tqdm(range(data_len)):
                index = y_train[i]
                if labelwise_data_count[i] > 0:
                    augment_count -= 1
                    labelwise_data_count[i] -= 1
                    image = self.augment_image(X_train(i))
                    list_X_train.append(list(image))
                    list_y_train.append(index)

        self.X_train = np.array(list_X_train)
        self.y_train = np.array(list_y_train)
        print("Shape of X-Train: {0} and y-train:{1}".format(X_train.shape, y_train.shape))

    def split_data(self, data):
        """Split data to training and validation set."""
        validation_file = Path("../data/valid.p")
        if validation_file.isfile():
            validate = []
            with open(validation_file, mode='rb') as f:
                validate = pickle.load(f)
            self.X_validate = validate['features']
            self.y_validate = validate['labels']
            self.X_validate.astype(np.float32)
            self.y_validate.astype(np.float32)
        else:
            self.X_train, self.y_train, self.X_validate, self.y_validate = train_test_split(data['features'],
                                                                            data['labels'], test_size=0.33,
                                                                            random_state=0)

    def one_hot_encoding(self):
        binarize = preprocessing.LabelBinarizer()
        binarize.fit(self.y_train)
        self.y_train    = binarize.transform(self.y_train)
        self.y_validate = binarize.transform(self.y_validate)
        self.y_test     = binarize.transform(self.y_test)
        print("One hot encoding done..")

    def read_and_process_data(self):
        """Load training and test data."""
        training_file = "../data/train.p"
        testing_file = "../data/test.p"

        with open(training_file, 'rb') as f:
            train = pickle.load(f)
        with open(testing_file, 'rb') as f:
            test = pickle.load(f)

        X_train, y_train = train['features'], train['labels']
        X_test, y_test = test['features'], test['labels']
        X_train.astype(np.float32)
        y_train.astype(np.float32)
        X_test.astype(np.float32)
        y_test.astype(np.float32)

        self.augment_data(X_train, y_train)
        self.one_hot_encoding()










