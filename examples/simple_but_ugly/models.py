# pylint: skip-file
import os
import sys
import threading
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import *


class MyModel:
    """An example of a model class """
    def __init__(self, mode, config=None):
        self.config = config
        print("\n\n___________________ MyModel initialized")

    def __call__(self, *args, **kwargs):
        print("MyModel call", args, kwargs)



# Example of custome Batch class which defines some actions
class MyBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)

    @model(mode='global')
    def global_model():
        print("Building a global model")
        with tf.variable_scope("global"):
            input_data = tf.placeholder('float', [None, 3])
            model_output = tf.square(tf.reduce_sum(input_data))
        return [input_data, model_output]

    @action(model='global_model')
    def train_global(self, model_spec):
        print("        action for a global model", model_spec)
        input_data, model_output = model_spec
        session = self.pipeline.get_variable("session")
        res = session.run(model_output, feed_dict={input_data: self.data})
        #print("        ", int(res))
        return self

    @model(mode='static')
    def static_model(pipeline):
        print("Building a static model")
        with pipeline.get_variable("session").graph.as_default():
            with tf.variable_scope("static"):
                input_data = tf.placeholder('float', [None, 3])
                model_output = tf.square(tf.reduce_sum(input_data))
        print("Static model is ready")
        return [input_data, model_output]

    @action(model='static_model')
    def train_static(self, model_spec):
        t1 = time()
        print("\n ================= train static ====================")
        input_data, model_output = model_spec
        session = self.pipeline.get_variable("session")
        t = time()
        res = session.run(model_output, feed_dict={input_data: self.data})
        te = time()
        print(te - t, te - t1)
        #print("        ", int(res))
        return self

    @model(mode='dynamic')
    def dynamic_model(self, config=None):
        print("Building a dynamic model with shape", self.data.shape)
        print("Model config", config)
        with self.pipeline.get_variable("session").graph.as_default():
            with tf.variable_scope("dynamic"):
                input_data = tf.placeholder('float', [None, self.data.shape[1]])
                model_output = tf.square(tf.reduce_sum(input_data))
        print("\n ***************** define dynamic *******************")
        print("----- default graph")
        print(tf.get_default_graph().get_operations())
        print()
        return [input_data, model_output]

    @action(use_lock='__train_dynamic')
    def train_dynamic(self):
        print("inside train")
        model_spec = self.get_model_by_name(MyBatch.dynamic_model)
        #print("        action for a dynamic model", model_spec)
        session = self.pipeline.get_variable("session")
        with self.pipeline.get_variable("print lock"):
            print("\n\n ================= train dynamic ====================")
            print("----- default graph")
            #print(tf.get_default_graph().get_operations())
            print("----- session graph")
            print(session.graph.get_operations())
        input_data, model_output = model_spec
        res = session.run(model_output, feed_dict={input_data: self.data})
        self.pipeline.get_variable("loss history").append(res)
        #print("        ", int(res))
        return self

    @action
    def test_dynamic(self):
        print("========== test dynamic =============")
        model_spec = self.get_model_by_name("dynamic_model")
        input_data, model_output = model_spec
        session = self.pipeline.get_variable("session")
        t = time()
        res = session.run(model_output, feed_dict={input_data: self.data})
        print(time() - t)
        print(int(res), self.data.sum() ** 2)
        return self

    @action
    def train_model(self, model_name):
        model = self.get_model_by_name(model_name)
        print("\n========== train external model =============")
        print("Train", model_name)
        return self


# number of items in the dataset
K = 100
Q = 10


# Fill-in dataset with sample data
def pd_data():
    ix = np.arange(K)
    data = np.arange(K * 3).reshape(K, -1).astype("float32")
    dsindex = DatasetIndex(ix)
    ds = Dataset(index=dsindex, batch_class=MyBatch)
    return ds, data.copy()


# Create datasets
ds_data, data = pd_data()

# Create tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

config = dict(dynamic_model=dict(arg1=0, arg2=0))


# Create a template pipeline
template_pp = (Pipeline(config=config)
                .init_variable("session", init=tf.Session)
                .init_variable("loss history", init=list, init_on_each_run=True)
                .init_model("static_model")
)

# Create another template
pp2 = (template_pp
        .init_variable("session", sess)
        .init_variable("print lock", init=threading.Lock)
        .define_model("dynamic", MyModel, "my_model")
        .define_model("dynamic", MyModel, "my_model2")
        #.init_model("MyModel")
        .load(data)
        #.train_global()
        .train_static()
        #.train_dynamic()
        .train_model("dynamic_model")
        #.train_model("MyModel")
        .train_model("my_model")
        .train_model("my_model2")
        .run(K//10, n_epochs=1, shuffle=False, drop_last=False, lazy=True)
)

# Create another template
t = time()
#res = (pp2 << ds_data).run()
print(time() - t)

print("-------------------------------------------")
print("============== start run ==================")
t = time()
res = (pp2 << ds_data).run()
print(time() - t)
#ModelDirectory.print()


print("-------------------------------------------------")
print("============== start gen_batch ==================")
print("Start iterating...")
res = pp2 << ds_data
t = time()
t1 = t
for batch in res.gen_batch(K, n_epochs=1, drop_last=True, prefetch=Q*0):
    with res.get_variable("print lock"):
        print("Batch", batch.indices, "is ready in", time() - t1)
    t1 = time()

print("Stop iterating:", time() - t)

#ModelDirectory.print()

print(res.get_variable("loss history"))

print(res.get_model_by_name("global_model"))

print(res.get_model_by_name("dynamic_model"))

res2 = (ds_data.pipeline()
               .init_variable("session", sess)
               #.import_model("dynamic_model", res)
               .load(data)
               #.test_dynamic()
)

print("--------------------------------------------")
print("============== start test ==================")
print(res2)
for batch in res2.gen_batch(3, n_epochs=1, drop_last=True, prefetch=Q*0):
    with res.get_variable("print lock"):
        print("Batch", batch.indices, "is ready in", time() - t1)
    t1 = time()

#print(res2.get_model_by_name("dynamic_model"))
