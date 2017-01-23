import json
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from glob import glob
from os.path import isfile, basename, isdir
from os import makedirs
from sklearn.utils import shuffle
from skimage.util import view_as_windows
from skimage.io import imread
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import numpy as np

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def missing_chunks(chink_size, data_folder, windows_folder):
    for batch_i, image_batch in enumerate(chunks(glob(data_folder), chink_size)):
        features_file = windows_folder + 'features_{}.npy'.format(batch_i)
        labels_files = windows_folder + 'labels_{}.npy'.format(batch_i)

        if not (isfile(features_file) and isfile(labels_files)):
            yield features_file, labels_files, image_batch


def get_feature_labels(train_files, img_shape):
    fish_species = set([path[13:-14] for path in train_files]) - {'NoF'}
    json_data = {fish: {} for fish in fish_species}

    for fish in fish_species:
        with open('./data/main_bounding_boxes/{}_labels.json'.format(fish.lower())) as bounding_box_file:
            fish_data = json.load(bounding_box_file)
            json_data[fish] = {
                f['filename']: [
                    (point['x'], point['y']) for point in f['annotations'] if point] for f in fish_data}

    features = []
    labels = []
    for train_file in train_files:
        filename = basename(train_file)
        fish = train_file[13:-14]
        # Check if it's a square, so the step can be set to img_shape[0]
        assert img_shape[0] == img_shape[1]
        step = img_shape[0]

        for row_i, columns in enumerate(view_as_windows(imread(train_file), img_shape, step)):
            for column_i, column in enumerate(columns):
                label = False
                if fish != 'NoF' and filename in json_data[fish] and json_data[fish][filename]:
                    points = np.array(json_data[fish][filename])/step
                    if points[0][0] <= row_i <= points[1][0] and points[0][1] <= column_i <= points[1][1]:
                        label = fish
                features.append(column[0])
                labels.append(label)

    return np.array(features, dtype=np.float32), labels


def maybe_cache_featurs_labels(img_shape):
    chunk_size = 500
    data_folder = './data/train/*/*.jpg'
    windows_folder = './data/windows/'

    if not isdir(windows_folder):
        makedirs(windows_folder)

    for features_file, labels_files, image_batch in tqdm(
            list(missing_chunks(chunk_size, data_folder, windows_folder)),
            unit='batch'):
        features, labels = get_feature_labels(image_batch, img_shape)
        features = features/255
        features, labels = shuffle(features, labels)

        # Save Features and Labels
        assert len(features) == len(labels)
        np.save(features_file, features)
        np.save(labels_files, labels)


def feature_labels_generator():
    windows_folder = './data/windows/'
    feature_label_paths = zip(
        sorted(glob(windows_folder + 'features_*.npy')),
        sorted(glob(windows_folder + 'labels_*.npy')))
    labeler = LabelBinarizer().fit(np.array([False, 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']))

    while True:
        for feature_path, label_path in feature_label_paths:
            features = np.load(feature_path)
            labels = labeler.transform(np.load(label_path))
            yield features, labels


def train(img_shape):
    # Model
    model = Sequential()

    model.add(
        Convolution2D(32, 3, 3, input_shape=img_shape, activation='relu', W_constraint=maxnorm(3), dim_ordering='tf'))
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 3, 3, activation='relu', W_constraint=maxnorm(3), dim_ordering='tf'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, activation='relu', W_constraint=maxnorm(3), dim_ordering='tf'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, activation='relu', W_constraint=maxnorm(3), dim_ordering='tf'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(8))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for features, labels in feature_labels_generator():
        model.fit(features, labels, nb_epoch=1)
    # TODO: Get generator to
    # samples_per_epoch = 100
    # model.fit_generator(feature_labels_generator(), samples_per_epoch, nb_epoch=10)

    return model


def run():
    img_shape = (200, 200, 3)
    maybe_cache_featurs_labels(img_shape)
    train(img_shape)


if __name__ == '__main__':
    run()
