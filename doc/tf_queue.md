# Tensorflow queues


To begin with, you need to define the queues which will serve as a source of data.
We are going to have 2 tensorflow blocks:
- some processing operations
- and a model (e.g. a neural network).

That is why we need two queues:
```python
preprocess_queue = tf.FIFOQueue(capacity=5)
premodel_queue = tf.FIFOQueue(capacity=5)
```

Let's define a preprocessing tensorflow graph
```python
input_tensor = preprocess_queue.dequeue()
step1_tensor = ... # some tf operation
step2_tensor = ... # more tf operations
preprocessed_tensor = ... # yet more operations
```

And here is a typical tensorflow model:
```python
next_batch_tensor = premodel_queue.dequeue()
model_output = ... # define your operations
cost = tf.reduce_mean(model_output)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

After we have defined a computational graph, we create and initialize a tensorflow session:
```python
sess = tf.Session()
sess.run(tf.initialize_all_variables())
```

Another important bit is a [batch class](batch.md):
```python
class MyBatch(Batch):
    def get_tensor(self):
        return np.asarray(self._data)

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
    def after_queue_action(self, tensor, session):
        # you don't need a feed_dict here, as data will be fed from a queue
        # you have to pass a session explicitly, as computations will be executed in a different thread
        self._data = tensor.eval(session=session)
        return self

    @action
    def one_more_action(self):
        # do something with self._data
        return self
```
Take a look at `get_tensor` method. It should return a numpy array that will be fed into a tensorflow queue.

Now we can define a [pipeline](pipeline.md):
```python
my_pipeline = my_dataset.p
                .load('/some/path')
                .preprocessing_action()
                .another_action()
                .tf_queue(queue=preprocess_queue, session=sess)
                .after_queue_action(preprocessed_tensor, session=sess)
                .one_more_action()
                .tf_queue(queue=premodel_queue, session=sess)
```
So we have used 2 queues:
- in the middle of the pipeline - after `another_action()`
- at the end of the pipeline - after the last action.

Also take into account, that since actions are executed in different threads, a tensorflow session should be stated explicitly.

```python
for i in range(MAX_ITER):
    batch = my_pipeline.next_batch(BATCH_SIZE, n_epochs=None, prefetch=5)
    # run one optimization step for the current batch
    sess.run([optimizer])
```
Note that we don't use `batch` here as it (in fact, `batch.get_tensor()`) was already put into the premodel queue.

### Attention!

1. There is not much point in using tensorflow queues without batch `prefetch`ing.
1. It can be beneficial to put `next_batch` and `sess.run([optimizer])` in different threads.
