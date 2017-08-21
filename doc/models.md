# Working with models

A batch class might also include model methods.

## Model definition method
A model definition method
- is marked with a `@model` decorator
- returns a model descriptor.

There are two modes of model definitions:
- static
- dynamic

Static model is compiled at a class compilation time, so even before any other code is run.
As a result, it has no access to any variable or code outside itself.

Dynamic model is compiled each time a pipeline is run, when some action requests a model descriptor.

```python
class MyArrayBatch(ArrayBatch):
    ...
    @model(mode='static')
    def basic_model():
        input_data = tf.placeholder('float', [None, 28])
        model_output = ...
        cost = tf.reduce_mean(tf.square(data - model_output))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        return [input_data, optimizer]

    @model(mode='dynamic')
    def advanced_model(self):
        with tf.Graph().as_default():
          input_data = tf.placeholder('float', (None,) + self.images.shape])
          model_output = ...
          cost = tf.reduce_mean(tf.square(data - model_output))
          optimizer = tf.train.AdamOptimizer().minimize(cost)
          session = tf.Session()
          session.run(tf.global_variable_initializer())
        return dict(session=session, input_data=input_data, train_step=optimizer)
```
It is for you to decide what the model descriptor is. It might be a list, dict or any other data sctructure, containing:
- TensorFlow placeholders, optimizers and other variables you need (e.g. a loss function value or a graph).
- a Keras model
- an mxnet module

or any other model specification you need.


Later you will get back this descriptor in a model-based actions method. So you have to include in it everything you need to train and evaulate the model.

Important notes:  
You should never call model definition methods. They are called internally.

## Model-based actions
After a model is defined, you might use it to train, evaluate or predict.

```python
class MyArrayBatch(ArrayBatch):
    ...
    @action(model='basic_model')
    def train_model(self, model_spec, session):
        input_data, optimizer = model_spec
        session.run([optimizer], feed_dict={input_data: self.data})
        return self
```
You add to an `@action` decorator an argument `model` with a model definition method name.

Train method might be added to a pipeline:
```python
full_workflow = my_dataset.p
                          .load('/some/path')
                          .some_preprocessing()
                          .some_augmentation()
                          .train_model(session=sess)
```
You do not need to pass a model into this action. The model is saved in an internal model directory and then passed to all actions based on this model.

You might have several actions based on the very same model.
```python
class MyArrayBatch(ArrayBatch):
    ...
    @action(model='basic_model')
    def train_model(self, model_spec):
        ...

    @action(model='basic_model')
    def evaluate_model(self, model_spec):
        ...

full_workflow = my_dataset.p
                          .init_variable("session", tf_session)
                          .load('/some/path')
                          .some_preprocessing()
                          .some_augmentation()
                          .train_model()
                          .evaluate_model()
```

## Parallel training
If you [prefetch](prefetch.md) with actions based on not-thread-safe models you might encounter that your model hardly learns anything. The reason is that model variables might not update concurrently. To solve this problem a lock can be added to an action to allow for only one concurrent execution:
```python
class MyBatch:
    ...
    @action(model='some_model', use_lock="some_model_lock")
    def train_it(self, model_spec):
        input_images, input_labels = model_spec[0]
        optimizer, cost, accuracy = model_spec[1]
        session = self.pipeline.get_variable("session")
        _, loss = session.run([optimizer, cost], feed_dict={input_images: self.images, input_labels: self.labels})
        return self
```
