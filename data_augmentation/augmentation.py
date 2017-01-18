from glob import glob
from random import randint, choice

from os.path import basename, exists

from os import makedirs
from skimage import io
from skimage.transform import rescale
from skimage.transform import rotate
from tqdm import tqdm
import numpy as np


def rand_rotate(image):
    angle = randint(1, 360)
    return rotate(image, angle, preserve_range=True).astype(np.uint8)


def rand_scale(image):
    scale = choice(np.arange(0.5, 0.9, 0.1))
    return rescale(image, scale, preserve_range=True).astype(np.uint8)


def augment(method, save_dir):
    train_filepaths = list(glob('./data/train/*/*.jpg'))
    labels = [path[13:-14] for path in train_filepaths]

    # Create label directories if they don't exist
    for label in labels:
        if not exists(save_dir + label):
            makedirs(save_dir + label)

    for file_path, label in tqdm(list(zip(train_filepaths, labels))):
        augmented_image = method(io.imread(file_path))
        io.imsave(save_dir + label + '/' + basename(file_path), augmented_image)


def run():
    augments = {
        'rotate': (rand_rotate, './data/augmentation/rotation/'),
        'scale': (rand_scale, './data/augmentation/scale/')
    }

    for name, arguments in augments.items():
        print('Augmenting {} images:'.format(name))
        augment(*arguments)


if __name__ == '__main__':
    run()
