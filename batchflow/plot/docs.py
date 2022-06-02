""" Documentation for `Plot`. """

GENERAL_DOC = """

    General idea is to display graphs and annotate them based on config of provided parameters for various
    `matplotlib` functions (e.g. `figsize` goes to figure initialization, `title` goes to `plt.set_title`, etc.).

    The logic behind the process is the following:
    1. Parse data — put provided arrays into double nested list. Nestedness levels define subplot and layer data order
       correspondingly. Also infer images combination mode — either overlay, separate or mixed.
    2. Parse figure axes if provided, else create them.
    3. Obtain default config for chosen plotting mode and update it with provided parameters.
    4. For every data item choose corresponding subplot and delegate data plotting to it.
    5. Annotate figure.
    6. Save plot.

    General parameters
    ----------
    data : np.ndarray, tuple or list
        If array, its dimensionality must match plot `mode`:
        - in 'image' mode 1d, 2d and 3d arrays are valid, thoug 3d image must be either 1- or 3- channeled;
        - in 'histogram' mode arrays of any dimensionality are valid, since they are flattened anyway;
        - in 'curve' and 'loss' modes 1d arrays are valid, defining polyline 'y' coordinates;
        If tuple, must contain two 1d arrays:
        - in 'curve' mode arrays define 'x' and 'y' polyline coordinates correspondingly;
        - in 'loss' mode array define 'y' coordinates of loss and learning rates popylines correspondingly;
        If list, must either contain arrays or tuples of format specified above. List might be either flat or nested.
        If list if flat, plotter parses data based on `combine` parameter value (see details below).
        If list is nested, outer level defines subplots order while inner one defines layers order.
    mode : 'image', 'histogram', 'curve', 'loss'
        If 'image' plot given arrays as images.
        If 'histogram' plot 1d histogram.
        If 'curve' plot given arrays as curve lines.
        If 'loss' plot given arrays as loss curves.
    combine : 'overlay', 'separate' or 'mixed'
        Whether overlay images on a single subplot, show them on separate ones or use mixed approach.
        Needs specifying only when `combine='separate'` required, since `combine='overlay'` is default and
        `combine='mixed'` is infered automatically from data (if data list is nested, no need specifiying `combine`).
    kwargs :
        - For one of `image`, `histogram`, `curve`, `loss` methods of `Layer` (depending on chosen mode).
            Parameters and data nestedness levels must match if they are lists meant for differents subplots/layers.
            Every param with 'image_', 'histogram_', 'curve_', 'loss_' prefix is redirected to corresponding method.
            See detailed parameters listings below.
        - For `annotate`.
            Every param with 'title_', 'suptitle_', 'xlabel_', 'ylabel_', 'xticks_', 'yticks_', 'xlim_', 'ylim_',
            colorbar_', 'legend_' or 'grid_' prefix is redirected to corresponding matplotlib method.
            Also 'facecolor', 'set_axisbelow', 'disable_axes' arguments are accepted.

    Notes on advanced parameters managing
    -------------------------------------
    Keep in mind, that set of parameters that are parsed by plotter directly is limited to ones most frequently used.
    However there is a way to provide any parameter to a specific plot method, using prefix corresponding to it.
    One must prepend that prefix to a parameter name itself and provide parameter value in argument under such name.

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `size` argument.
    Providing `{'size': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.

    See specific prefices examples in sections below.

    Parameters for figure creation
    ------------------------------
    figsize : tuple
        Size of displayed figure. If not provided, infered from data shapes.
    facecolor : string or tuple of 3 or 4 numbers
        Figure background color. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    dpi : float
        The resolution of the figure in dots-per-inch.
    ncols, nrows : int
        Number of figure columns/rows.
    tight_layout : bool
        Whether adjust subplot parameters using `plt.tight_layout` with default padding or not. Defaults is True.
    figure_{parameter} : misc
        Any parameter valid for `plt.subplots`. For example, `figure_sharex=True`.
"""

IMAGE_DOC = """
    Parameters for 'image' mode
    ---------------------------
    transpose: tuple
        Order of axes for displayed images.
    dilate : bool, int, tuple of two ints or dict
        Parameter for image dilation via `cv2.dilate`.
        If bool, indicates whether image should be dilated once with default kernel (`np.ones((1,3))`).
        If int, indcates how many times image should be dilate with default kernel.
        If tuple of two ints, defines shape of kernel image should be dilate with.
        If dict, must contain keyword arguments for `cv2.dilate`.
    mask : number, str, callable or tuple of any of them
        Parameter indicating which values should be masked.
        If a number, mask this value in data.
        If str, must consists of operator and a number (e.g. '<0.5', '==2', '>=1000').
        If a callable, must return boolean mask with the same shape as original image that mark image pixels to mask.
        If a tuple, contain any combination of items of types above.
    cmap : str or matplotlib colormap object
        Сolormap to display single-channel images with. Must be valid matplotlib colormap (e.g. 'ocean', 'tab20b').
    mask_color : string or tuple of 3 or 4 numbers
        Color to display masked values with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    alpha : number in (0, 1) range
        Image opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    vmin, vmax : number
        Limits for normalizing image into (0, 1) range. Values beyond range are clipped (default matplotlib behaviour).
    extent : tuple of 4 numbers
        The bounding box in data coordinates that the image will fill.
    image_{parameter} : misc
        Any parameter valid for `Axes.imshow`. For example, `image_interpolate='bessel'`.
"""

HISTOGRAM_DOC = """
    Parameters for 'histogram' mode
    -------------------------------
    flatten : bool
        Whether convert input array to 1d before plot. Default is True.
    mask : number, str, callable or tuple of any of them
        Parameter indicating which values should be masked.
        If a number, mask this value in data.
        If str, must consists of operator and a number (e.g. '<0.5', '==2', '>=1000').
        If a callable, must return boolean mask with the same shape as original image that mark image pixels to mask.
        If a tuple, contain any combination of items of types above.
    color : string or tuple of 3 or 4 numbers
        Color to display histogram with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    alpha : number in (0, 1) range
        Histogram opacity (0 means fully transparent, i.e. invisible, and 1 - totally opaque).
        Useful when `combine='overlay'`.
    bins : int
        Number of bins for histogram.
    histogram_{parameter} : misc
        Any parameter valid for `Axes.hist`. For example, `histogram_density=True`.
"""

CURVE_DOC = """
    Parameters for 'curve' mode
    ---------------------------
    color : string or tuple of 3 or 4 numbers
        Color to display curve with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    linestyle : str
        Style to display curve with. Must be valid matplotlib line style (e.g. 'dashed', ':').
    alpha : number in (0, 1) range
        Curve opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    curve_{parameter} : misc
        Any parameter valid for `Axes.plot`. For example, `curve_marker='h'`.
"""

LOSS_DOC = """
    Parameters for 'loss' mode
    ----------------------------
    color : string or tuple of 3 or 4 numbers
        Color to display loss curve with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    linestyle : str
        Style to display loss curve with. Must be valid matplotlib line style (e.g. 'dashed', ':').
    linewidth : number
        Width of loss curve.
    alpha : number in (0, 1) range
        Curve opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    window : None or int
        Size of the window to use for moving average calculation of loss curve.
    loss_{parameter}, lr_{parameter} : misc
        Any parameter valid for `Axes.plot`. For example, `curve_fillstyle='bottom'`.
"""

ANNOTATION_DOC = """
    Parameters for axes annotation
    ------------------------------
    label : str
        Text that should be put on legend against patches corresponding to layers objects.
    suptitle, title, xlabel, ylabel or {text_object}_label: str
        Text that should be put in corresponding annotation object.
    {text_object}_color : str, matplotlib colormap object or tuple
        Color of corresponding text object. Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
        If str, ust be valid matplotlib colormap.
        Must be valid matplotlib color (e.g. 'roaylblue', '#120FA3', (0.3, 0.4, 0.5)).
    {text_object}_size : number
        Size of corresponding text object. Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
    colorbar : bool
        Toggle for colorbar.
    colorbar_width : number
        The width of colorbar as a percentage of the subplot width.
    colorbar_pad : number
        The pad of colorbar as a percentage of the subplot width.
    legend_loc : number
        Codes legend position in matplotlib terms (must be from 0-9 range).
    grid: bool
        Grid toggle.
    {object}_{parameter} : misc
        Any parameter with prefix of desired object that is valid for corresponding method:
        - 'text_' — for every text object (title, label etc.)
        - 'title_' — for `Axes.set_title`
        - 'xlabel_' — for `Axes.set_xlabel`
        - 'ylabel_' — for `Axes.set_ylabel`
        - 'xticks_' — for `Axes.set_xticks`
        - 'yticks_' — for `Axes.set_yticks`
        - 'tick_' — for `Axes.tick_params`
        - 'xlim_' — for `Axes.set_xlim`
        - 'ylim_' — for `Axes.set_ylim`
        - 'colorbar_' — for `Axes.colorbar`
        - 'legend_' — for `Axes.legend`
        - 'minor_grid_', 'major_grid_' — `Axes.grid`
"""

EXAMPLES_DOC = """
    Data display scenarios
    ----------------------
    1. The simplest one if when one provide a single data array — in that case data is displayed on a single subplot:
       >>> Plot(array)
    2. A more advanced use case is when one provide a list of arrays — plot behaviour depends on `combine` parameter:
       a. Images are put on same subplot and overlaid one over another if `combine='overlay'` (which is default):
          >>> Plot([image_0, mask_0])
       b. Images are put on separate subplots if `combine='separate'`.
          >>> Plot([image_0, image_1], combine='separate')
    3. The most complex scenario is displaying images in a 'mixed' manner (ones overlaid and others separated).
       For example, to overlay first two images but to display the third one separately, use the following notation:
       >>> Plot([[image_0, mask_0], image_1]); (`combine='mixed'` is set automatically if data is double-nested).

    The order of arrays inside the double-nested structure basically declares, which of them belong to the same subplot
    and therefore should be rendered one over another, and which must be displayed separately.

    If a parameter is provided in a list, each subplot uses its item on position corresponding to its index and
    every subplot layer in turn uses item from that sublist on positions that correspond to its index w.r.t. to subplot.
    Therefore, such parameters must resemble data nestedness level, since that allows binding subplots and parameters.
    However, it's possible for parameter to be a single item — in that case it's shared across all subplots and layers.

    For example, to display two images separately with same colormap, the following code required:
    >>> Plot([image_0, image_1], cmap='viridis')
    If one wish to use different colormaps for every image, the code should be like this:
    >>> Plot([image_0, image_1], cmap=['viridis', 'magma'])
    Finally, if a more complex data provided, the parameter nestedness level must resemble the one in data:
    >>> Plot([[image_0, mask_0], [image_1, mask_1]], cmap=[['viridis', 'red'], ['magma', 'green']])
"""
