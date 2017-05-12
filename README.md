# Dataset

## Basic usage

`Dataset` helps you conveniently work with random or sequential batches of your data:
```python
NUM_ITERS = 1000
for batch in my_dataset.gen_batch(BATCH_SIZE, shuffle=False, n_epochs=1):
    # ...
```
and define processing workflows even for datasets that do not fit into memory:
```python
my_workflow = my_dataset.pipeline()
              .load('/some/path')
              .do_something()
              .do_something_else()
              .some_additional_action()
              .save('/to/other/path')
```
The trick here is that all the processing actions are lazy. They are not executed until their results are needed, e.g. when you request a preprocessed batch:
```python
for i in range(NUM_ITERS):
    processed_batch = my_workflow.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
    # only now the actions are fired and data is changed with the workflow defined earlier
```

For more advanced cases and detailed API see [the documentation](doc/intro.md)


## Installation

### Git submodule
```
git submodule add https://github.com/analysiscenter/dataset.git
git submodule init
git submodule update
```
