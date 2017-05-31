# Tensorflow queues

A pipeline might send data directly into Tensorflow queues.


To begin with, you need a [batch class](batch.md):
```python
class MyBatch(Batch):
    def get_tensor(self):
        return np.asarray(self.data)

    @action
    def load(path):
        self._data = ...

    @action
    def preprocessing_action(self):
        # do something with self._data
        return self

    @action
    def another_action(self):
        # do something else with self._data
        return self

    @action
    def one_more_action(self):
        # do something with self._data
        return self
```
Take a look at `get_tensor` method. It should return a numpy array that will be fed into a tensorflow queue.

Create a queue:
```python
input_queue = tf.FIFOQueue(capacity=5)
```

Define a tensorflow model:
```python
next_batch_tensor = input_queue.dequeue()
model_output = ... # define your operations
cost = tf.reduce_mean(model_output)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

Here comes a [pipeline](pipeline.md) definition:
```python
my_pipeline = my_dataset.p
                .load('/some/path')
                .preprocessing_action()
                .another_action()
                .one_more_action()
                .tf_queue(queue=input_queue, get_tensor=MyBatch.get_tensor)
```
So we have defined a queue at the end of the pipeline - after the last action.

After we have defined batch actions, a computational graph and a pipeline, we create and initialize a tensorflow session:
```python
sess = tf.Session()
sess.run(tf.initialize_all_variables())
```


And now let's iterate over batches and train the model:
```python
for i in range(MAX_ITER):
    batch = my_pipeline.next_batch(BATCH_SIZE, n_epochs=None, prefetch=5, tf_session=sess)
    # run one optimization step for the current batch
    sess.run([optimizer])
```
Note that we don't use `batch` here as it was already put into the input queue (in fact, not the batch itself, but `batch.get_tensor()`).

Also since actions are executed in different threads, a tensorflow session should be stated explicitly when calling pipeline's `next_batch`, `gen_batch` or `run` methods.

### Attention!

1. There is not much point in using tensorflow queues without batch `prefetch`ing.
1. It can be beneficial to put `next_batch` and `sess.run([optimizer])` in different threads.
