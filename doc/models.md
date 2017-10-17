# Working with models

Pipelines can include model definitions, training, evauluation and prediction actions which links models and pipelines.

## Content
1. [Model class](#model-class)
1. [Model types](#model-types)
1. [Adding a model to a pipeline](#adding-a-model-to-a-pipeline)
1. [Configuring a model](#configuring-a-model)
1. [Training a model](#training-a-model)
1. [Predicting with a model](#predicting-with-a-model)
1. [Saving a model](#saving-a-model)
1. [Models and template pipelines](#models-and-template-pipelines)
1. [Importing models](#importing-models)
1. [Parallel training](#parallel-training)
1. [Writing your own model](#writing-your-own-model)


## Model class
Models are defined in model classes. You can use standard models or write your own models.

```python
from dataset.models import BaseModel

class MyModel(BaseModel):
    def _build(self, *args, **kwargs):
        #... do whatever you want
```
See below [Writing your own model](#writing-your-own-model).


## Model types

There are two modes of model definition:
- static
- dynamic

A static model is created when the pipeline is created, i.e. before the pipeline is run.
As a result, the model might have an access to the pipeline, its config and variables.

A dynamic model is created during the first pipeline's execution, when some action uses the model the first time.
Consequently, the model has an access to the pipeline and the very first batch thus allowing to create models adapting to shapes and data sizes.


## Adding a model to a pipeline
First of all, a model should be initialized:
```python
full_workflow = my_dataset.p
                          .init_model('static', MyModel, 'my_model', config)
                          ...
```
In `init_model` you state a mode (`static` or `dynamic`), a model class, an optional short model name (otherwise, a class name will be used) and an optional configuration.
A static model is initialized immediately in the `init_model`, while a dynamic model will be initialized when the pipeline is run and the very first batch flows into the pipeline.

If a model was already created in another pipeline, it might be [imported](#importing-models).


## Configuring a model
Most often than not models have many options and hyperparameters which define the model structure (e.g. number and types of layers for a neural network or a number and depth of trees for forests) or training procedure (e.g. an optimization algorithm or regularization constants).

Global options:
- `build` : bool - whether to call `model.build(...)` to create a model. Default is `True`.
- `load` : bool - whether to load a model from a persistent storage.  
If `load=True`, then `model.load()` will be called. Default is `False`.  
Loading usually requires some additional config parameters like paths, file names or file formats.

For some models only one of `build` or `load` can be `True`. While other models might need a building phase even if a model is loaded from a disk.

Read a model specfication to know how to configure it.


## Training a model
A train action should be stated below an initialization action:
```python
full_workflow = my_dataset.p
                          .init_model('static', MyModel, 'my_model', config)
                          ...
                          .train_model('my_model', x='images', y='labels')
```
`train_model`'s arguments might be specific to a particular model you use. So read a model specfication to find out what it expects for training.

Model independent arguments are:
- `make_data` - a function or method which takes a current batch and a model instance and return a dict of arguments for `model.train(...)`.
- `save_to` - a batch component name or a pipeline variable name to store an output of `model.train` (if there is any).  
If both (a batch component and a pipeline variable with the same name) exist, then a batch component will be used. So be careful with naming.
- `append_to` - a pipeline variable name where a model output will be appended to.  
If both (`save_to` and `append_to`) are present, then only `append_to` will be used.

```python
full_workflow = my_dataset.p
                          .init_model('static', MyModel, 'my_model', my_config)
                          .init_model('dynamic', AnotherModel, 'another_model', another_config)
                          .init_variable('current_accuracy', 0)
                          .init_variable('loss_history', init_on_each_run=list)
                          ...
                          .train_model('my_model', output='accuracy', x='images', y='labels',
                                       save_to='current_accuracy')
                          .train_model('another_model', fecthes='loss',
                                       feed_dict={'x': ''images', 'y': ''labels'},
                                       append_to='loss_history')
```

You can also write an action which works with a model directly.
```python
class MyBatch(Batch):
    ...
    @action(model='some_model')
    def train_linked_model(self, model):
        ...

    @action
    def train_in_batch(self, model_name):
        model_spec = self.get_model_by_name(model_name)
        ...


full_workflow = my_dataset.p
                          .init_model('static', MyModel, 'my_model', my_config)
                          .init_model('dynamic', MyOtherModel, 'some_model', some_config)
                          .some_preprocessing()
                          .some_augmentation()
                          .train_in_batch('my_model')
                          .train_linked_model()
```


## Predicting with a model
`predict_model` is very similar to [`train_model`](#training-a-model) described above:
```python
full_workflow = my_dataset.p
                          .init_model('static', MyModel, 'my_model', config)
                          .init_variable('predicted_labels', init_on_each_run=list)
                          ...
                          .predict_model('my_model', x='images', save_to='predicted_labels')
```
Read a model specfication to find out what it needs for predicting and what its output is.


## Saving a model
You can write a model to a persistent storage at any time by calling `save_model(...)`
```python
some_pipeline.save_model('my_model', path='/some/path')
```
As usual, the first argument is a model name, while all other arguments are model specific, so read a model documentation
to find out what parameters are required to save a model.

Note, that `save_model` is imperative, i.e. it saves a model right now, but not when a pipeline is executed.
Thus, it cannot be a part of a pipeline's chain of actions (otherwise, this would save the model after processing each batch,
which might be highly undesired).

`save_model` is expected to be called in an action method or after a training pipeline has finished
(e.g. after [`pipeline.run`](pipeline.md#running-pipelines)).


## Models and template pipelines
A template pipeline is not linked to any dataset and thus it will never run. It might be used as a building block for more complex pipelines.

```python
template_pipeline = Pipeline().
                       .init_model('static', MyModel)
                       .init_model('dynamic', MyModel2)
                       .prepocess()
                       .normalize()
                       .train_model('MyModel', some_transform_fn)
                       .train_model('MyModel2', some_transform_fn2)
```

Linking a pipeline to a dataset creates a new pipeline that can be run.
```python
mnist_pipeline = (template_pipeline << mnist_dataset).run(BATCH_SIZE, n_epochs=10)
cifar_pipeline = (template_pipeline << cifar_dataset).run(BATCH_SIZE, n_epochs=10)
```
Take into account, that a static model is created only once in the template_pipeline.
But it will be used in each children pipeline with different datasets (which might be a good or bad thing).

Whilst, a separate instance of a dynamic model will be created in each children pipeline.


## Importing models
Models exist within pipelines. This is very convenient if a single pipeline includes everything: preprocessing, model training, model evaluation, model saving and so on. However, sometimes you might want to share a model between pipelines. For instance, when you train a model in one pipeline and later use it in an inference pipeline.

This can be easily achieved with a model import.

```python
train_pipeline = (images_dataset.p
                       .init_model('dynamic', Resnet50)
                       .load(...)
                       .random_rotate(angle=(-30, 30))
                       .train_model("Resnet50")
                       .run(BATCH_SIZE, shuffle=True, n_epochs=10))

inference_pipeline_template = (Pipeline()
                                  .resize(shape=(256, 256))
                                  .normalize()
                                  .import_model("Resnet50", train_pipeline)
                                  .predict_model("Resnet50"))
...

infer = (inference_pipeline_template << some_dataset).run(INFER_BATCH_SIZE, shuffle=False)
```
When `inference_pipeline_template` is run, the model `Resnet50` from `train_pipeline` will be imported.


## Parallel training
If you [prefetch](prefetch.md) with actions based on non-thread-safe models you might encounter that your model hardly learns anything. The reason is that model variables might not update concurrently. To solve this problem a lock can be added to an action to allow for only one concurrent execution:
```python
class MyBatch:
    ...
    @action(use_lock="some_model_lock")
    def train_it(self, model_spec):
        input_images, input_labels = model_spec[0]
        optimizer, cost, accuracy = model_spec[1]
        session = self.pipeline.get_variable("session")
        _, loss = session.run([optimizer, cost], feed_dict={input_images: self.images, input_labels: self.labels})
        return self
```
However, as far as `tensorflow` is concerned, its optimizers have a parameter [`use_locking`](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#__init__) which allows for concurrent updates when set to `True`.


## Writing your own model
