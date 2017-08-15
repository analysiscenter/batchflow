# pylint: skip-file
import os
import sys
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import *


# Example of custome Batch class which defines some actions
class MyArrayBatch(ArrayBatch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)

    @model(mode='static')
    def static_model():
        print("Building a static model")
        input_data = tf.placeholder('float', [None, 3])
        model_output = tf.square(tf.reduce_sum(input_data))
        return [input_data, model_output]

    @action(model='static_model')
    def train_static(self, model, session):
        print("        action for a static model", model)
        input_data, model_output = model
        res = session.run(model_output, feed_dict={input_data: self.data})
        print("        ", int(res))
        return self

    @model(mode='dynamic')
    def dynamic_model(self):
        print("Building a dynamic model with shape", self.data.shape)
        input_data = tf.placeholder('float', [None, 3])
        model_output = tf.square(tf.reduce_sum(input_data))
        return [input_data, model_output]

    @action(model='dynamic_model')
    def train_dynamic(self, model, session):
        print("        action for a dynamic model", model)
        input_data, model_output = model
        res = session.run(model_output, feed_dict={input_data: self.data})
        print("        ", int(res))
        return self


# number of items in the dataset
K = 100
Q = 10


# Fill-in dataset with sample data
def pd_data():
    ix = np.arange(K)
    data = np.arange(K * 3).reshape(K, -1).astype("float32")
    dsindex = DatasetIndex(ix)
    ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
    return ds, data.copy()


# Create datasets
ds_data, data = pd_data()

# Create tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create pipeline
res = (ds_data.pipeline()
        .load(data)
        .train_static(sess)
        .train_dynamic(sess)
)

print("Start iterating...")
t = time()
t1 = t
for batch in res.gen_batch(3, n_epochs=1, drop_last=True, prefetch=Q*2*0):
    print("Batch", batch.indices, "is ready in", time() - t1)
    t1 = time()

print("Stop iterating:", time() - t)
