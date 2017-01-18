from glob import glob
from random import randint

from os.path import basename, exists

from os import makedirs
from skimage import io
from skimage.transform import rotate
from tqdm import tqdm
import numpy as np


def rand_rotate(image):
    angle = randint(1, 360)
    rotated_image = rotate(image, angle)

    # Fix Warning of float64 to uint8 conversion in skimage.io.imsave
    rotated_image *= 255
    rotated_image = rotated_image.astype(np.uint8)

    return rotated_image


def run():
    rotation_folder = './data/augmentation/rotation/'
    train_filepaths = list(glob('./data/train/*/*.jpg'))
    labels = [path[13:-14] for path in train_filepaths]

    # Create label directories if they don't exist
    for label in labels:
        if not exists(rotation_folder + label):
            makedirs(rotation_folder + label)

    for file_path, label in tqdm(list(zip(train_filepaths, labels))):
        augmented_image = rand_rotate(io.imread(file_path))
        io.imsave(rotation_folder + label + '/' + basename(file_path), augmented_image)


if __name__ == '__main__':
    run()
