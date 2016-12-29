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
from os.path import basename

import numpy as np

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


def preprocess(image_shape, image_paths, labels=[]):
    features = []
    for image_path in tqdm(image_paths):
        image_data = list(Image.open(image_path).resize(image_shape[:2]).getdata())
        image_data = np.asarray(image_data).reshape(image_shape)

        features.append(image_data)

    # Normalizer
    features = np.asarray(features)
    features = features / 255.0

    if labels:
        # one hot encode
        label_binarizer = LabelBinarizer()
        labels = label_binarizer.fit_transform(labels)
        # Shuffle
        features, labels = shuffle(features, labels)

    return features, labels


def get_featurs_labels(image_shape):
    data_folder = './data/train/*/*.jpg'
    features_file = 'features.npy'
    labels_files = 'labels.npy'

    if not (isfile(features_file) and isfile(labels_files)):
        train_files = list(glob(data_folder))
        labels = [path[13:-14] for path in train_files]

        features, labels = preprocess(image_shape, train_files, labels)

        # Save Features and Labels
        np.save(features_file, features)
        np.save(labels_files, labels)
    else:
        features = np.load(features_file)
        labels = np.load(labels_files)

    return features, labels


def train(img_shape):
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
    return model


def predict(img_shape, model):
    prediction_file = 'predictions.csv'
    predict_folder = './data/test_stg1/*.jpg'
    image_paths = list(glob(predict_folder))

    features, _ = preprocess(img_shape, image_paths)
    predictions = model.predict(features)

    # Save file
    header = 'image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT'
    with open(prediction_file, 'w') as out:
        out.write(header + '\n')
        for image_paths, prediction in zip(image_paths, predictions):
            out.write(basename(image_paths) + ',' + ','.join([str(x) for x in prediction]) + '\n')


def run():
    img_shape = (80, 45, 3)

    model = train(img_shape)
    predict(img_shape, model)


if __name__ == '__main__':
    run()
