import numpy as np
from PIL import Image
import os
import glob
import random
from scipy.signal import convolve2d
from skimage.util import random_noise


# Some functions defined below were taken from drill exercise 3 of the course.
def read_img(path, mono=False):
    if mono:
        return read_img_mono(path)
    img = Image.open(path)
    return np.asarray(img)


def read_img_mono(path):
    # The L flag converts it to 1 channel.
    img = Image.open(path).convert(mode="L")
    return np.asarray(img)


def resize_img(ndarray, size):
    # Parameter "size" is a 2-tuple (width, height).
    img = Image.fromarray(ndarray.clip(0, 255).astype(np.uint8))
    return np.asarray(img.resize(size))


def save_img(ndarray, path):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).save(path)


# Used the number of images of the category with the least images, for both train and test
def get_image_paths(data_path, categories, num_train_per_cat=307, num_test_per_cat=54):
    '''
    This function returns lists containing the file path for each train
    and test image, as well as lists with the label of each train and
    test image.
    '''

    num_categories = len(categories)  # number of categories.

    train_image_paths = [None] * (num_categories * num_train_per_cat)
    test_image_paths = [None] * (num_categories * num_test_per_cat)

    train_labels = [None] * (num_categories * num_train_per_cat)
    test_labels = [None] * (num_categories * num_test_per_cat)

    for i, cat in enumerate(categories):
        images = glob.glob(os.path.join(data_path, 'train', cat, '*.jpg'))

        for j in range(num_train_per_cat):
            train_image_paths[i * num_train_per_cat + j] = images[j]
            train_labels[i * num_train_per_cat + j] = cat

        images = glob.glob(os.path.join(data_path, 'test', cat, '*.jpg'))
        for j in range(num_test_per_cat):
            test_image_paths[i * num_test_per_cat + j] = images[j]
            test_labels[i * num_test_per_cat + j] = cat
    return train_image_paths, test_image_paths, train_labels, test_labels


def get_test_image_paths(data_path, categories, num_test_per_cat=54):
    '''
    This function returns lists containing the file path for each test image,
    as well as lists with the label of each test image, for perturbation purposes.
    '''

    num_categories = len(categories)  # number of categories.

    test_image_paths = [None] * (num_categories * num_test_per_cat)

    test_labels = [None] * (num_categories * num_test_per_cat)

    for i, cat in enumerate(categories):
        images = glob.glob(os.path.join(data_path, cat, '*.jpg'))
        for j in range(num_test_per_cat):
            test_image_paths[i * num_test_per_cat + j] = images[j]
            test_labels[i * num_test_per_cat + j] = cat
    return test_image_paths, test_labels


# Helper functions for perturbations
def gaussian_pixel_noise(img, sd):
    shape = img.shape
    noise = np.random.normal(0, sd, shape)
    noised_image = img + noise

    if len(img.shape) == 2:
        grayscale = True
    else:
        grayscale = False

    if not grayscale:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if noised_image[i][j][k] < 0:
                        noised_image[i][j][k] = 0
                    if noised_image[i][j][k] > 255:
                        noised_image[i][j][k] = 255
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                if noised_image[i][j] < 0:
                    noised_image[i][j] = 0
                if noised_image[i][j] > 255:
                    noised_image[i][j] = 255
    return np.asarray(noised_image)


def gaussian_blurring(img, times):
    mask = [[1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16]]
    convolved_image = img.copy()

    if len(img.shape) == 2:
        grayscale = True
    else:
        grayscale = False

    if not grayscale:
        for i in range(times):
            convolved_image[:, :, 0] = convolve2d(convolved_image[:, :, 0], mask, mode='same')
            convolved_image[:, :, 1] = convolve2d(convolved_image[:, :, 1], mask, mode='same')
            convolved_image[:, :, 2] = convolve2d(convolved_image[:, :, 2], mask, mode='same')
    else:
        for i in range(times):
            convolved_image = convolve2d(convolved_image, mask, mode='same')

    return convolved_image


def image_contrast_change(img, scale):
    shape = img.shape
    changed_image = img * scale

    if len(img.shape) == 2:
        grayscale = True
    else:
        grayscale = False

    if not grayscale:
        if scale > 1:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        if changed_image[i][j][k] > 255:
                            changed_image[i][j][k] = 255
    else:
        if scale > 1:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if changed_image[i][j] > 255:
                        changed_image[i][j] = 255
    return np.asarray(changed_image)


def image_brightness_change(img, change):

    if len(img.shape) == 2:
        grayscale = True
    else:
        grayscale = False

    if not grayscale:
        if change > 0:
            shape = img.shape
            changed_image = img + change
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        if changed_image[i][j][k] > 255:
                            changed_image[i][j][k] = 255
        elif change < 0:
            shape = img.shape
            changed_image = img + change
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        if changed_image[i][j][k] < 0:
                            changed_image[i][j][k] = 0
        else:
            changed_image = img
    else:
        if change > 0:
            shape = img.shape
            changed_image = img + change
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if changed_image[i][j] > 255:
                        changed_image[i][j] = 255
        elif change < 0:
            shape = img.shape
            changed_image = img + change
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if changed_image[i][j] < 0:
                        changed_image[i][j] = 0
        else:
            changed_image = img

    return np.asarray(changed_image)


def occlusion(img, length):
    shape = img.shape
    occluded_image = img.copy()
    random_row = random.randint(0, shape[0]-1-length)
    random_column = random.randint(0, shape[1]-1-length)
    occluded_image[random_row:random_row+length, random_column:random_column+length] = 0
    return np.asarray(occluded_image)


def salt_and_pepper_noise(img, strength):
    noised_img = random_noise(img, mode='s&p', amount=strength)
    return np.asarray(255*noised_img)
