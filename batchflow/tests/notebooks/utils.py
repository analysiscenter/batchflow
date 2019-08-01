""" Utility functions for model debugging. """
import matplotlib.pyplot as plt



BAD_OPS = [
    '/globals/', '/Initializer/', '/batch_normalization/',
    '/gradients/', 'Adam', 'microbatch',
    'mean_squared_error', 'softmax_cross_entropy_loss',
]


def print_model_graph(pipeline, start=0, depth=-1, bad_ops=None):
    """ Show model graph without some of the junk. """
    bad_ops = bad_ops or BAD_OPS
    model = pipeline.get_model_by_name('MODEL')

    for item in model.graph.get_operations():
        flag = sum([op in item.name for op in bad_ops])
        if flag == 0:
            name_list = item.name.split('/')
            name_list = name_list[start:]
            name_list = name_list[:depth]
            name = '/'.join(name_list)
            print(name)


def plot_loss(pipeline):
    """ Get loss graph. """
    plt.figure(figsize=(10, 5))
    plt.plot(pipeline.v('loss_history'))

    plt.xlabel('Iterations')
    plt.ylabel('loss', fontdict={'fontsize': 15})
    plt.grid(True)

    loss_name = pipeline.config['model_config']['loss']
    plt.title('{} loss'.format(loss_name), fontdict={'fontsize': 15})
    plt.show()
