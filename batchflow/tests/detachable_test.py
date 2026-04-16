"""Test that the detachable decorator and Plot.save work with detach=True.

The original implementation used multiprocessing.Process which fails on Python 3.14
due to a pickling error: `@wraps(func)` preserves `__qualname__`, so pickle resolves
`Plot.plot` by name and finds the wrapper instead of the original function.

Replacing Process with Thread avoids pickling entirely — figure saving doesn't need
process isolation.
"""

import os
import tempfile

import numpy as np


def test_detachable_plot_with_detach():
    """Plot.plot with detach=True should complete without PicklingError."""
    from batchflow.plotter.plot import Plot

    data = np.random.rand(10, 10)
    with tempfile.TemporaryDirectory() as tmpdir:
        savepath = os.path.join(tmpdir, "test_detach.png")
        p = Plot(data=data, mode="image", show=False, detach=True, savepath=savepath)
        # detach=True runs in a daemon thread — give it a moment to finish
        import time
        time.sleep(1)
        # The plot object should have been created without error
        assert p is not None


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
        import time
        deadline = time.monotonic() + 5
        while not os.path.exists(savepath2) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert os.path.exists(savepath2)


def test_detachable_plot_without_detach():
    """Plot.plot with detach=False (default) should work as before."""
    from batchflow.plotter.plot import Plot

    data = np.random.rand(10, 10)
    with tempfile.TemporaryDirectory() as tmpdir:
        savepath = os.path.join(tmpdir, "test_no_detach.png")
        Plot(data=data, mode="image", show=False, savepath=savepath)
        assert os.path.exists(savepath)
