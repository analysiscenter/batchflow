# pylint: skip-file
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import action, model
from dataset.image import ImagesBatch
from dataset.opensets import MNIST
from dataset.models.tf import TFModel
from dataset.models.tf.layers import conv2d_block, iflatten


class MyModel(TFModel):
    def _build(seld, *args, **kwargs):
        input_images = tf.placeholder("uint8", [None, 28, 28, 1], name='images')
        input_labels = tf.placeholder("uint8", [None], name='labels')
        images = tf.to_float(input_images)

        features = conv2d_block(images, 32, (3, 3), layout='canp', name='layer1')
        features = iflatten(features)

        layer1 = tf.layers.dense(features, units=512, activation=tf.nn.relu)
        model_output = tf.layers.dense(layer1, units=10)
        model_output = tf.identity(model_output, name='predictions')

        encoded_labels = tf.one_hot(input_labels, depth=10, name='targets')
        predicted_labels = tf.argmax(model_output, axis=1, name='predicted_labels')


if __name__ == "__main__":
    BATCH_SIZE = 256 * 4

    mnist = MNIST()

    print()
    print("Start training...")
    train_pp = (mnist.train.p
                .init_variable('loss_history', init_on_each_run=list)
                .init_variable('current_loss', init_on_each_run=0)
                .init_model('static', MyModel, 'conv', config={'loss': 'ce'})
                .train_model('conv', fetches='loss', feed_dict={'images': 'images', 'labels': 'labels'}, save_to='current_loss')
                .append_variable('loss_history', 'current_loss')
                .print_variable('current_loss')
                .run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True))
    print("End training")

    print()
    print("Start testing...")
    test_pp = (mnist.test.p
                .import_model('conv', train_pp)
                .init_variable('all_predictions', init_on_each_run=list)
                .init_variable('predictions')
                .predict_model('conv', fetches='predicted_labels', feed_dict={'images': 'images', 'labels': 'labels'}, save_to='predictions')
                .append_variable('all_predictions', 'predictions')
                .run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=False))
    print("End testing")

    print("Predictions")
    for pred in test_pp.get_variable("all_predictions"):
        print(pred.shape)

    conv = train_pp.get_model_by_name("conv")
