# Transfer Learning with GoogLeNet
## Retain
First do to the tensorflow source directory in `submodule/tensorflow`, then build TensorFlow.
```sh
./configure
```
Build the retrainer
```sh
bazel build -c opt --copt=-mavx tensorflow/examples/image_retraining:retrain
```
Retrain on the training data.

_To see all the hyperparameters, check out the [retrain file](https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/examples/image_retraining/retrain.py)._
```sh
bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir <TRAIN_DATSET_DIR>
```
