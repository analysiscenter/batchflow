# pylint: skip-file
import os
import sys
import threading
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import *
from dataset.models import TFModel


class MyModel(TFModel):
    """An example of a tf model class """
    def _build(self, *args, **kwargs):
        num_features = 3
        num_classes = 3
        x = tf.placeholder("float", [None, num_features], name='x')
        y = tf.placeholder("int32",[None], name='y')
        y_oe = tf.one_hot(y, num_classes)

        w = tf.Variable(tf.zeros([num_features,num_classes]))
        b = tf.Variable(tf.zeros([num_classes]))

        y_ = tf.nn.softmax(tf.matmul(x, w) + b)

        # Define a cost function
        tf.losses.add_loss(tf.losses.softmax_cross_entropy(y_oe, y_))

        print("___________________ MyModel initialized")


class MyBatch(Batch):
    components = 'images', 'labels'

    @action(model='static_model')
    def train_in_batch(self, model_spec):
        print("train in batch model", model_spec)
        return self


def trans(batch):
    return dict(feed_dict=dict(x=batch.data[:, :-1], y=batch.data[:, -1].astype('int')))

# number of items in the dataset
K = 100
Q = 10


# Fill-in dataset with sample data
def gen_data():
    ix = np.arange(K)
    data = np.arange(K * 3).reshape(K, -1).astype("float32")
    labels = np.random.choice(3, size=K).astype("int32")
    dsindex = DatasetIndex(ix)
    ds = Dataset(index=dsindex, batch_class=MyBatch)
    return ds, data, labels


# Create datasets
ds_data, data, labels = gen_data()

# Create tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

config = dict(dynamic_model=dict(arg1=0, arg2=0))


# Create a template pipeline
pp = (Pipeline(config=config)
        .init_model("static", MyModel, name="static_model")
        .init_model("dynamic", MyModel, name="dynamic_model")
        .load((data, labels))
        #.train_model("static_model", fn=trans)
        .train_in_batch()
        .train_model("dynamic_model", feed_dict={'x': 'images', 'y': 'labels'})
        .run(K//10, n_epochs=1, shuffle=False, drop_last=False, lazy=True)
)

# Create another template
t = time()
#res = (pp2 << ds_data).run()
print(time() - t)

print("-------------------------------------------")
print("============== start run ==================")
t = time()
res = (pp << ds_data).run()
print(time() - t)
