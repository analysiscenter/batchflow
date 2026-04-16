"""Test that the detachable decorator and Plot.save work with detach=True.

The original implementation used multiprocessing.Process which fails on Python 3.14
due to a pickling error: `@wraps(func)` preserves `__qualname__`, so pickle resolves
`Plot.plot` by name and finds the wrapper instead of the original function.

Replacing Process with Thread avoids pickling entirely — figure saving doesn't need
process isolation.
"""

import os
import tempfile
import time

import numpy as np


def _wait_for_file(path, timeout=5):
    """Poll until `path` exists or `timeout` seconds elapse."""
    deadline = time.monotonic() + timeout
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.05)


def test_detachable_plot_with_detach():
    """Plot.plot with detach=True should complete without PicklingError."""
    from batchflow.plotter.plot import Plot

    data = np.random.rand(10, 10)
    with tempfile.TemporaryDirectory() as tmpdir:
        savepath = os.path.join(tmpdir, "test_detach.png")
        p = Plot(data=data, mode="image", show=False, detach=True, savepath=savepath)
        _wait_for_file(savepath)
        assert p is not None
        assert os.path.exists(savepath)


def test_plot_save_with_detach():
    """Plot.save with detach=True should complete without PicklingError."""
    from batchflow.plotter.plot import Plot

    data = np.random.rand(10, 10)
    with tempfile.TemporaryDirectory() as tmpdir:
        savepath = os.path.join(tmpdir, "test_save_detach.png")
        p = Plot(data=data, mode="image", show=False, savepath=savepath)
        assert os.path.exists(savepath)

        savepath2 = os.path.join(tmpdir, "test_save_detach2.png")
        p.save(savepath=savepath2, detach=True)
        _wait_for_file(savepath2)
        assert os.path.exists(savepath2)


def test_detachable_plot_without_detach():
    """Plot.plot with detach=False (default) should work as before."""
    from batchflow.plotter.plot import Plot

    data = np.random.rand(10, 10)
    with tempfile.TemporaryDirectory() as tmpdir:
        savepath = os.path.join(tmpdir, "test_no_detach.png")
        Plot(data=data, mode="image", show=False, savepath=savepath)
        assert os.path.exists(savepath)
