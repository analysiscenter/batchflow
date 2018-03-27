# pylint: disable=too-many-locals
import matplotlib.pyplot as plt
import seaborn as sns

def plot_weights(model_names, model_weights, model_params, colors, num_axis, num_blocks, bottleneck=True):
    """Plot distribution of weights

    Parameters
    ----------
    model_names : list or str
        name layers of model

    model_weights : list
        all weights of model

    model_params : list
        number of parameters in layers

    colors : list
        names of colors

    num_axis : list with two elements
        [nrows, ncols] in plt.subplots

    bottleneck : bool
        use bottleneck

    num_blocks : list
        numbers of blocks to draw
        """
    nrows, ncols = num_axis
    _, subplot = plt.subplots(nrows, ncols, sharex='all', figsize=(23, 24))
    subplot = subplot.reshape(-1)
    num_plot = 0
    dict_names = {'bottleneck': {'layer-1': 'first conv 1x1',
                                 'layer-4': 'conv 3x3',
                                 'layer-7': 'second conv 1x1'},
                  'no_bottle': {'layer-1': 'first conv 3x3',
                                'layer-4': 'second conv 3x3'}}

    bottle = 'bottleneck' if bottleneck else 'no_bottle'

    for names, weights, num_params in separate(model_names, model_weights, model_params, bottleneck, num_blocks):
        for name, weight, num in zip(names, weights, num_params):

            if name != 'shortcut' and name != 0:
                name = dict_names[bottle][name]

            subplot[num_plot].set_title('Number of parameners={}\n{}'.format(num, name), fontsize=18)

            if not isinstance(weight, int):
                sns.distplot(weight.reshape(-1), ax=subplot[num_plot], color=colors[int(num_plot % ncols)])

                if num_plot % 1 == 0:
                    dis = (6. / ((weight.shape[2] + weight.shape[3]) * weight.shape[0] * weight.shape[1])) ** 0.5
                    subplot[num_plot].axvline(x=dis, ymax=10, color='k')
                    subplot[num_plot].axvline(x=-dis, ymax=10, color='k')

            subplot[num_plot].set_xlabel('value', fontsize=20)
            subplot[num_plot].set_ylabel('quantity', fontsize=20)
            num_plot += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)