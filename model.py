from glob import glob
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from tqdm import tqdm
from os.path import isfile
from sklearn.utils import shuffle
from os.path import basename

import numpy as np
import matplotlib.pyplot as plt

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

GEN_FOLDER = './gen'

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


def get_features_labels(image_shape):
    
    data_folder = './data/train/*/*.jpg'
    features_file = GEN_FOLDER + '/features.npy'
    labels_files = GEN_FOLDER + '/labels.npy'

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

def plot_history(history):
    
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(GEN_FOLDER + '/accuracy.png', bbox_inches='tight')
    plt.pause(0.001)
    plt.ion()
    plt.show()
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(GEN_FOLDER + '/loss.png', bbox_inches='tight')

def train(img_shape):
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    # Model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=img_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))
    model.summary()

    features, labels = get_features_labels(img_shape)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # checkpoint
    filepath = GEN_FOLDER + "/top.model.weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    
    # tensorboard visualizaton
    # in a new terminal run: $ tensorboard --logdir=./logs
    # then open tensorboard in browser: http://localhost:[port_shown_in_terminal]
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    
    callbacks = [checkpoint, early_stopping, tensorboard]
    
    # fit the model
    history = model.fit(features, labels, nb_epoch=2, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=1)
    
    # plot graphs
    plot_history(history)
    
    return model


def predict(img_shape, model):
    
    predict_folder = './data/test_stg1/*.jpg'
    
    prediction_file = GEN_FOLDER + '/ predictions.csv'
    validations_file = GEN_FOLDER + '/validations.npy'
    
    image_paths = list(glob(predict_folder))

    if not (isfile(validations_file)):
        features, _ = preprocess(img_shape, image_paths)
        
        # Save to disk
        np.save(validations_file, features)
    else:
        features = np.load(validations_file)
        
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
