# Working with models

A batch class might also include model methods.

## Model definition method
A model definition method
- is marked with a `@model` decorator
- returns a model descriptor.

There are two modes of model definitions:
- global
- static
- dynamic

A global model is compiled at a class compilation time, so even before any other code is run.
As a result, it has no access to any variable or code outside itself even to `self` argument.

A static model exists within a pipeline and is compiled when `init_model(...)` is added to the pipeline (so before the pipeline is run).
As a result, it has an access to the pipeline and its config.

A dynamic model exists within a pipeline and is compiled each time the pipeline is run, when some action requests a model descriptor.
Consequently, it has access to everything else (including a batch `self` argument) thus allowing to build models adapting to shapes and data sizes.

```python
class MyBatch(Batch):
    ...
    @model(mode='global')
    def basic_model():
        input_data = tf.placeholder('float', [None, 28, 28, 1])
        output_data = tf.placeholder('float', [10])
        model_output = ...
        cost = tf.reduce_mean(tf.square(output_data - model_output))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        return [input_data, optimizer]

    @model(mode='static')
    def static_model(pipeline, config=None):
        with pipeline.get_variable("session").graph.as_default():
            with tf.variable_scope("static"):
                input_data = tf.placeholder('float', [None, 28, 28, 1])
                output_data = tf.placeholder('float', [10])
                model_output = ...
                cost = tf.reduce_mean(tf.square(output_data - model_output))
                optimizer = tf.train.AdamOptimizer().minimize(cost)
        return [input_data, optimizer]

    @model(mode='dynamic')
    def advanced_model(self, config=None):
        with tf.Graph().as_default():
            input_data = tf.placeholder('float', (None,) + self.images.shape])
            output_data = tf.placeholder('float', [10])
            model_output = ...
            cost = tf.reduce_mean(tf.square(output_data - model_output))
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

**Important note:**  
You should never call model definition methods. They are called internally.

## Model-based actions
After a model is defined, you might use it to train, evaluate or predict.

```python
class MyBatch(Batch):
    ...
    @action(model='basic_model')
    def train_model(self, model_spec, session):
        input_data, optimizer = model_spec
        session.run([optimizer], feed_dict={input_data: self.data})
        return self
```
You add to an `@action` decorator an argument `model` with a model definition method name.

A train method might be added to a pipeline:
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
class MyBatch(Batch):
    ...
    @action(model='basic_model')
    def train_model(self, model_spec):
        session = self.pipeline.get_variable("session")
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

You don't need to write train and test methods for every model as there is a convenient `get_model_by_name` function:
```python
class MyBatch(Batch):
    ...
    @action
    def train_model(self, model_name):
        model_spec = self.get_model_by_name(model_name)
        ...

my_pipeline = my_dataset.p
                 .one_action()
                 .another_action()
                 .train_model("vgg19")
                 .train_model("resnet50")
```

## Static model initialization
Static models exist within pipelines, but before the pipeline is run. As a consequence, you should explicitly declare which static models you need in the pipeline.
```python
template_pipeline = Pipeline().
                       .init_model("my_static_model")
                       .prepocess()
                       .normalize()
```
This is a template pipeline and it will never run. It is used as a building block for more complex pipelines.

```python
my_mnist_pipeline = (template_pipeline << mnist_dataset).run(BATCH_SIZE, n_epochs=10)
my_cifar_pipeline = (template_pipeline << cifar_dataset).run(BATCH_SIZE, n_epochs=10)
```
`my_static_model` will be defined only once in the `init_model(...)`.
But it will be used many times in the each children pipeline with different datasets.
That is why static models do not have access to data shapes (since they may differ in differen datasets)


## Model import
Dynamic and static models exist within pipelines. This is not a problem if a single pipeline includes everything: preprocessing, model training, model evaluation, model saving and so on. However, sometimes you might want to share a model between pipelines. For instance, when you train a model in one pipeline and later use it in an inference pipeline.

This can be easily achieved with a model import.

```python
train_pipeline = (images_dataset.p
                       .load(...)
                       .random_rotate(angle=(-30, 30))
                       .train_classifier(model_name="resnet50")
                       .run(BATCH_SIZE, shuffle=True, n_epochs=10)

inference_pipeline_template = (Pipeline()
                                  .resize(shape=(256, 256))
                                  .normalize()
                                  .import_model("resnet50", train_pipeline)
                                  .get_prediction(model_name="resnet50")
)
```
When `inference_pipeline_template` is run, the model descriptor `resnet50` from `train_pipeline` will be imported.

For this to work `images_dataset`'s batch class should contain an action `train_classifier` and [a model definition method](#model-definition-method) named "resnet50".


## Parallel training
If you [prefetch](prefetch.md) with actions based on non-thread-safe models you might encounter that your model hardly learns anything. The reason is that model variables might not update concurrently. To solve this problem a lock can be added to an action to allow for only one concurrent execution:
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
