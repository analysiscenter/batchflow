# pylint: skip-file
import os
import sys
import threading
from time import time
import numpy as np
import tensorflow as tf

sys.path.append("../..")
from dataset import *
from dataset.models import BaseModel


class MyModel(BaseModel):
    """An example of a model class """
    def build(self, *args, **kwargs):
        print("____ MyModel initialized")

    def train(self, *args, **kwargs):
        return 1, 2, 3



# Example of custome Batch class which defines some actions
class MyBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)

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
    def train_in_batch(self, model_name):
        print("\n========== train external model =============")
        model = self.get_model_by_name(model_name)
        print("Train", model_name)
        return self

    def make_data_for_dynamic(self):
        return {'shape': self.data.shape}



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
                .init_variable("loss history", init=list, init_on_each_run=True)
                .init_model("static_model", MyModel)
)

# Create another template
pp2 = (template_pp
        .init_variable("print lock", init=threading.Lock)
        .init_model("dynamic", MyModel, "my_model", config=F(MyBatch.make_data_for_dynamic))
        .init_model("dynamic", MyModel, "my_model2")
        .load(data)
        .train_model("my_model")
        .train_model("my_model2", save_to=V('output'))
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


print("-------------------------------------------------")
print("============== start gen_batch ==================")
res = pp2 << ds_data
print("Start iterating...")
t = time()
t1 = t
for batch in res.gen_batch(K, n_epochs=1, drop_last=True, prefetch=Q*0):
    with res.get_variable("print lock"):
        print("Batch", batch.indices, "is ready in", time() - t1)
    t1 = time()

print("Stop iterating:", time() - t)



pp3 = (Pipeline()
           .import_model("my_model2", res)
           .load(data)
           .train_model("my_model2")
)

print("--------------------------------------------")
print("============== start test ==================")
res2 = pp3 << ds_data
for batch in res2.gen_batch(3, n_epochs=1, drop_last=True, prefetch=Q*0):
    with res.get_variable("print lock"):
        print("Batch", batch.indices, "is ready in", time() - t1)
    t1 = time()

res3 = pp3 << ds_data
print("predict")
res3.run(3, n_epochs=1)
