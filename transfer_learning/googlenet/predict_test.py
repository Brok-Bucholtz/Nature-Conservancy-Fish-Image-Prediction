from glob import glob
import numpy as np
import tensorflow as tf
from helper import save_predictions
from tqdm import tqdm


def create_graph(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        tf.import_graph_def(graph_def, name='')


def run():
    model_path = '/tmp/output_graph.pb'
    prediction_file = 'predictions.csv'
    image_paths = list(glob('./data/test_stg1/*.jpg'))
    image_predictions = []

    with open('/tmp/output_labels.txt', 'rb') as labels_file:
        labels = [line.decode("utf-8").upper() for line in labels_file.read().splitlines()]

    with tf.Session() as sess:
        create_graph(model_path)
        for image_path in tqdm(image_paths):
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            image_predictions.append(predictions)

    save_predictions(prediction_file, image_paths, labels, image_predictions)


if __name__ == '__main__':
    run()
