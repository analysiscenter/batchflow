# pylint: skip-file
import os
import sys
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import action, model, B, C, F, V
from dataset.image import ImagesBatch
from dataset.opensets import MNIST
from dataset.models.tf import TFModel
from dataset.models.tf.layers import conv2d_block, flatten


class MyModel(TFModel):
    def _build(self, *args, **kwargs):
        images_shape = [None] + list(self.get_from_config('images_shape'))

        input_images = tf.placeholder("uint8", images_shape, name='input_images')
        input_labels = tf.placeholder("uint8", [None], name='input_labels')
        images = tf.to_float(input_images)

        features = conv2d_block(images, 32, 3, layout='canp', name='layer1')
        features = flatten(features)

        layer1 = tf.layers.dense(features, units=512, activation=tf.nn.relu)
        model_output = tf.layers.dense(layer1, units=10)
        predictions = tf.identity(model_output, name='predictions')

        targets = tf.one_hot(input_labels, depth=10, name='targets')
        predicted_labels = tf.argmax(model_output, axis=1, name='predicted_labels')


if __name__ == "__main__":
    BATCH_SIZE = 256

    mnist = MNIST()

    print()
    print("Start training...")
    t = time()
    train_pp = (mnist.train.p
                .init_variable('loss_history', init_on_each_run=list)
                .init_variable('current_loss', init_on_each_run=0)
                .init_variable('pred_label', init_on_each_run=list)
                .init_variable('input_tensor_name', 'input_images')
                .init_model('dynamic', MyModel, 'conv',
                            config={'session': {'config': tf.ConfigProto(allow_soft_placement=True)},
                                    'loss': 'ce',
                                    'optimizer': {'name':'Adam', 'use_locking': True},
                                    'images_shape': F(lambda batch: batch.images.shape[1:])})
                .train_model('conv', fetches=['loss', 'predicted_labels'],
                                     feed_dict={V('input_tensor_name'): B('images'),
                                                'input_labels': B('labels')},
                             save_to=[V('current_loss'), V('pred_label')])
                .print_variable('current_loss')
                .update_variable('loss_history', V('current_loss'), mode='a'))

    train_pp.run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0)
    #train_pp.next_batch(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0)
    print("End training", time() - t)


    print()
    print("Start testing...")
    t = time()
    test_pp = (mnist.test.p
                .import_model('conv', train_pp)
                .init_variable('all_predictions', init_on_each_run=list)
                .predict_model('conv', fetches='predicted_labels', feed_dict={'input_images': B('images'),
                                                                              'input_labels': B('labels')},
                               append_to=V('all_predictions'))
                .run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=False, prefetch=4))
    print("End testing", time() - t)

    print("Predictions")
    for pred in test_pp.get_variable("all_predictions"):
        print(pred.shape)

    conv = train_pp.get_model_by_name("conv")
