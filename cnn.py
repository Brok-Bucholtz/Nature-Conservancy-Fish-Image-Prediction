from glob import glob
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from tqdm import tqdm
from os.path import isfile
from sklearn.utils import shuffle
from sklearn.preprocessing import Normalizer

import numpy as np

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


def get_featurs_labels(image_shape):
    data_folder = './data/train/*/*.jpg'
    features_file = 'features.npy'
    labels_files = 'labels.npy'

    if not (isfile(features_file) and isfile(labels_files)):
        # load data
        features = []
        labels = []
        for path in tqdm(list(glob(data_folder))):
            image_data = list(Image.open(path).resize(image_shape[:2]).getdata())
            image_data = np.asarray(image_data).reshape(image_shape)

            features.append(image_data)
            labels.append(path[13:-14])

        # Normalizer
        normalizer = Normalizer()
        features = normalizer.fit_transform(features)

        # one hot encode
        label_binarizer = LabelBinarizer()
        labels = label_binarizer.fit_transform(labels)
        features, labels = shuffle(features, labels)

        # Save Features and Labels
        np.save(features_file, features)
        np.save(labels_files, labels)
    else:
        features = np.load(features_file)
        labels = np.load(labels_files)

    return features, labels


def train():
    img_shape = (80, 45, 3)
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    # Model
    model = Sequential()
    model.add(Convolution2D(
        32, 3, 3, input_shape=img_shape, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    features, labels = get_featurs_labels(img_shape)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(features, labels, nb_epoch=10, batch_size=32, validation_split=0.2, verbose=1)


if __name__ == '__main__':
    train()
