from glob import glob
from random import randint

from os.path import basename, exists

from os import makedirs
from skimage import io
from skimage.transform import rotate
from tqdm import tqdm
import numpy as np


def run():
    rotation_folder = './data/augmentation/rotation/'
    train_filepaths = list(glob('./data/train/*/*.jpg'))
    labels = [path[13:-14] for path in train_filepaths]

    for file_path, label in tqdm(list(zip(train_filepaths, labels))):
        angle = randint(1, 360)
        train_file = io.imread(file_path)
        rated_image = rotate(train_file, angle)

        # Create directory if it doesn't exist
        if not exists(rotation_folder + label):
            makedirs(rotation_folder + label)

        # Fix Warning of float64 to uint8 conversion in skimage.io.imsave
        rated_image *= 255
        rated_image = rated_image.astype(np.uint8)

        io.imsave(rotation_folder + label + '/' + basename(file_path), rated_image)


if __name__ == '__main__':
    run()
