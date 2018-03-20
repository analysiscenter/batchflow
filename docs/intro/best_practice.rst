=============
Best practice
=============

Machine learning models often imply numerous hyperparameters which impact model performance and training.
These are include optimization algorithms, layer layouts, batch norm momentum and other parameters.

After training thousands of models we find out that some values of those parameters work better than others.
That is why we gathered them into :doc:`best practice <best_practice>` module.
Though it does not fit all the situations and sometimes can even lead to average results, most of the time it works well.


Optimizer
=========
use_locking = True


Batch normalization
===================
momentum = .1


VGG
===
head = layout: 'Vdf', dropout_rate=.8


ResNet
======
convolution layers layout = 'cna'
