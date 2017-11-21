# pylint: skip-file
import os
import sys
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import Pipeline, B, C, F, V
from dataset.opensets import MNIST
from dataset.models.tf import TFModel, VGG16, VGG19, VGG7, FCN32, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                              Inception_v1, Inception_v3, SqueezeNet, MobileNet

class MyModel(TFModel):
    def _build(self, config=None):
        tf.losses.add_loss(1.)
        pass

if __name__ == "__main__":
    BATCH_SIZE = 4

    mnist = MNIST()

    train_template = (Pipeline()
                .init_variable('loss_history', init_on_each_run=list)
                .init_variable('current_loss', init_on_each_run=0)
                .init_variable('pred_label', init_on_each_run=list)
                .init_model('dynamic', MyModel, 'conv',
                            config={'inputs': dict(images={'shape': B('image_shape')},
                                                   labels={'classes': 10, 'transform': 'ohe', 'name': 'targets'}),
                                    'input_block/inputs': 'images',
                                    'head/units': [100, 100, 10],
                                    'loss': None,
                                    'optimizer': None,
                                    #'filters': 16, 'width_factor': 1,
                                    #'body': dict(se_block=1, se_factor=4, resnext=1, resnext_factor=4, bottleneck=1),
                                    'output': dict(ops=['labels', 'accuracy'])})
                .resize(shape=(128, 128))
                .train_model('conv', #fetches='loss',
                                     feed_dict={'images': B('images'),
                                                'labels': B('labels')},
                             save_to=V('current_loss'), use_lock=True)
                .print_variable('current_loss')
                .update_variable('loss_history', V('current_loss'), mode='a'))

    train_pp = (train_template << mnist.train)
    print("Start training...")
    t = time()
    train_pp.run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=False, prefetch=0)
    print("End training", time() - t)


    print()
    print("Start testing...")
    t = time()
    test_pp = (mnist.test.p
                .import_model('conv', train_pp)
                .init_variable('accuracy', init_on_each_run=list)
                .predict_model('conv', fetches='accuracy', feed_dict={'images': B('images'),
                                                                      'labels': B('labels')},
                               save_to=V('accuracy'), mode='a')
                .run(BATCH_SIZE, shuffle=True, n_epochs=1, drop_last=True, prefetch=0))
    print("End testing", time() - t)

    accuracy = np.array(test_pp.get_variable('accuracy')).mean()
    print('Accuracy {:6.2f}'.format(accuracy))


    conv = train_pp.get_model_by_name("conv")
