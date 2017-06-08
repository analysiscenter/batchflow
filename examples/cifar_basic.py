# pylint: skip-file
import os
import sys
import numpy as np
import tensorflow as tf
from time import time
import threading

sys.path.append("..")
from dataset.opensets import CIFAR10


if __name__ == "__main__":
    BATCH_SIZE = 64
    N_ITERS = 1000

    cifar = CIFAR10()
    N_CLASSES = len(np.unique(cifar._train_labels))


    input_images = tf.placeholder("uint8", [None, 32, 32, 3])
    input_labels = tf.placeholder("uint8", [None])

    encoded_labels = tf.one_hot(input_labels, depth=N_CLASSES)
    input_vectors = tf.cast(tf.reshape(input_images, [-1, 32 * 32 * 3]), 'float')
    layer1 = tf.layers.dense(input_vectors, units=1024, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, units=512, activation=tf.nn.relu)
    layer3 = tf.layers.dense(layer1, units=256, activation=tf.nn.relu)
    model_output = tf.layers.dense(layer3, units=N_CLASSES)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=encoded_labels, logits=model_output))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    prediction = tf.argmax(model_output, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(encoded_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    print("Start training...")
    t = time()
    i = 0
    for batch in cifar.train.gen_batch(BATCH_SIZE, shuffle=False, n_epochs=2):
        i += 1
        _, loss = sess.run([optimizer, cost], feed_dict={input_images: batch.images, input_labels: batch.labels})
        if (i + 1) % 50 == 0:
            print("Iteration", i + 1, "loss =", loss)
    print("Iteration", i + 1, "loss =", loss)
    print("End training", time() - t)

    print()
    print("Start validating...")
    for i in range(3):
        batch = cifar.test.next_batch(BATCH_SIZE * 10, shuffle=False, n_epochs=None)
        acc = sess.run(accuracy, feed_dict={input_images: batch.images, input_labels: batch.labels})
        print("Batch", i, "accuracy =", acc)
    print("End validating\n")
