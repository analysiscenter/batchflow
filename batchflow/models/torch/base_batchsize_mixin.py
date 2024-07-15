""" Contains mixin for :class:`~.torch.TorchModel` to compute optimal batch size. """
import gc

import numpy as np
import torch

from ...monitor import GPUMemoryMonitor
from ...notifier import Notifier



class OptimalBatchSizeMixin:
    """ Compute optimal batch size for training/inference to maximize GPU memory usage. """

    def is_oom_error(self, exception):
        """ Check whether exception is OOM error """
        if not (isinstance(exception, RuntimeError) and len(exception.args) == 1):
            return False
        return (
            "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
            or "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
            or ("CUDA" in exception.args[0] and "out of memory" in exception.args[0])
        )

    def garbage_collection_cuda(self):
        """ Garbage collection Torch (CUDA) memory. """
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exception:
            if not self.is_oom_error(exception):
                raise


    def compute_optimal_batch_size(self, method='train', inputs=None, targets=None,
                                   start_batch_size=4, max_iters=25, factor=2,
                                   spread=0.2, estimation_method='bruteforce', n=4,
                                   max_memory=100, pbar='n', tail_size=20,
                                   frequency=0.05, delta_batch_size=16,
                                   max_batch_size=1024, max_iters_estimation=2):
        """ Compute optimal batch size in two steps:
        1. Calculate batch size estimation using `predictive` or `bruteforce` method.
        2. Calculate exact optimal batch size using binary search.

        Parameters
        ----------
        method: str, {`train`, `predict`}
            Defines in which method (`train` or `predict`) the optimal batch size will
            be computed. Default: `train`.

        estimation_method: str, {`bruteforce`, `predictive`}
            Whether `bruteforce` or `predictive` estimation method will be used.

        inputs: np.ndarray
            The inputs to the model, that will be used in optimal batch computation.
            If not provided, then the placeholder will be created

        targets: np.array
            The targets to the model, that will be used in optimal batch computation.
            If not provided, then we run model.predict() to get targets. Only used in method=`train`

        start_batch_size: int
            Batch size to start batch_size estimation. If your model is small, you should use larger
            start_batch_size in order to get accurate optimal batch_size

        max_iters: int
            Maximum number of binary search iterations.

        factor: int
            Value by which we multiply `batch_size` at each iteration. Used in bruteforce estimation.

        spread: float
            Used to create an interval for binary search.
            The interval is ((1 - `spread`) * `batch_size_estimation`; (1 + `spread`) * `batch_size_estimation`)
            Used in `predictive` estimation.

        n: int
            For stable measurements, we make `n` iterations of `train`/`predict`,
            until the memory consumption stabilizes. Used only in estimation_method='predictive'.

        max_memory: int
            In percent. Defines which portion of memory will be occupied by the model with optimal batch size.
            Default: 100

        pbar: str
            The same as bar in Notifier

        tail_size: int
            How many items of gpu memory data will be used to compute mean memory utilization.

        frequency: int
            How often do we collect gpu memory data.

        delta_batch_size: int
            Step size of the change in batch size at each iteration.

        max_batch_size: int
            Maximum batch_size

        max_iters_estimation: int
            Maximum number of batch size estimation iterations.

        Returns
        -------
        optimal_batch_size: int
            Batch size that fits in `max_memory`
        """

        # first calculate optimal batch_size estimation
        if estimation_method == 'predictive':
            batch_size_estimation = self._compute_optimal_batch_size_predictive(method=method, max_memory=max_memory,
                                                                                inputs=inputs, targets=targets,
                                                                                start_batch_size=start_batch_size,
                                                                                delta_batch_size=delta_batch_size,
                                                                                max_iters=max_iters_estimation,
                                                                                max_batch_size=max_batch_size,
                                                                                pbar='n', n=n)
            batch_size_estimation = batch_size_estimation['batch_size']
            low = int(batch_size_estimation * (1 - spread))
            high = int(batch_size_estimation * (1 + spread))
        elif estimation_method == 'bruteforce':
            low = self._compute_optimal_batch_size(inputs=inputs, targets=targets, factor=factor,
                                                   start_batch_size=start_batch_size, method=method,
                                                   max_memory=max_memory, pbar=pbar, tail_size=10,
                                                   frequency=frequency, update_method='bruteforce')
            high = low * factor
            batch_size_estimation = (low + high) // 2
        else:
            raise ValueError("Wrong estimation method! It could be `predictive` or `bruteforce`.")

        # then run precise method in neighbourhood of batch_size_estimation
        return self._compute_optimal_batch_size(inputs=inputs, targets=targets,
                                                start_batch_size=batch_size_estimation,
                                                low=low, high=high, method=method,
                                                max_iters=max_iters, tail_size=tail_size,
                                                max_memory=max_memory, pbar=pbar,
                                                frequency=frequency, factor=factor,
                                                update_method='binary')


    def _bruteforce_batch_size_generator(self, factor, max_memory):
        """ Calculates next batch size for bruteforce estimation method. If consumed memory is lower
        than max_memory, then batch_size is multiplied by factor, otherwise it is divided by factor

        Yields
        ------
        new_batch_size, exit: tuple(int, bool)
            New batch size to check, and exit condition whether the optimal
            batch size computation is finished
        """

        while True:
            batch_size, consumed_memory = yield

            if consumed_memory > max_memory:
                yield batch_size // factor, True
            else:
                yield batch_size * factor, False


    def _binary_batch_size_generator(self, low, high, max_memory):
        """ Calculates next batch size for binary search method. If consumed memory is lower
        than max_memory, then lower bound is increased, otherwise the upped bound is decreased.

        Yields
        ------
        new_batch_size, exit: tuple(int, bool)
            New batch size to check, and exit condition whether the optimal
            batch size computation is finished.
        """
        while True:
            batch_size, consumed_memory = yield

            if consumed_memory > max_memory:
                high = batch_size
            else:
                low = batch_size

            exit = high - low <= 1
            yield (high + low) // 2, exit


    def _compute_optimal_batch_size(self, inputs=None, targets=None, low=2, high=None,
                                    start_batch_size=4, max_iters=15, pbar='n',
                                    method='train', max_memory=100, frequency=0.01,
                                    tail_size=20, update_method='bruteforce',
                                    factor=2):
        count = 0
        # if None => make equal distance between low, start_batch_size and high
        high = high if high is not None else 2 * start_batch_size - low

        # The list is used to show current batch_size in notifier. Current value is the last one.
        batch_size = start_batch_size if isinstance(start_batch_size, list) else [start_batch_size]

        if update_method == 'binary':
            n_iters = int(np.ceil(np.log2(high - low)))
            generator = self._binary_batch_size_generator(low=low, high=high, max_memory=max_memory)

        elif update_method == 'bruteforce':
            n_iters = None
            generator = self._bruteforce_batch_size_generator(factor=factor, max_memory=max_memory)
        else:
            raise ValueError("Wrong update method! Could be `bruteforce` or `binary`")

        notifier = Notifier(n_iters=n_iters, bar=pbar,
                            monitors=[{'source': batch_size, 'name': 'batch_size'}])

        while True:
            try:
                notifier.update()
                count += 1

                # monitor consumed memory
                with GPUMemoryMonitor(frequency=frequency) as monitor:
                    input = inputs or self.make_placeholder_data(batch_size[-1], to_device=False)
                    input = list(input) if isinstance(input, (tuple, list)) else [input]
                    input = [item[:batch_size[-1]] for item in input]

                    if method == 'train':
                        target = targets or self.predict(inputs=input, outputs='predictions')
                        target = list(target) if isinstance(target, (tuple, list)) else [target]
                        target = [item[:batch_size[-1]] for item in target]

                        _ = self.train(inputs=input, targets=target, microbatch_size=False)
                    else:
                        _ = self.predict(inputs=input, microbatch_size=False)

                data = monitor.data
                # take mean of top_k memory measures
                consumed_memory = np.mean(np.sort(data, axis=0)[-tail_size:])

                next(generator)
                generator.send(batch_size[-1])
                new_batch_size, exit = generator.send(consumed_memory)
                batch_size.append(new_batch_size)
            except RuntimeError as exception:
                if self.is_oom_error(exception):
                    next(generator)
                    generator.send(batch_size[-1])
                    new_batch_size, exit = generator.send(max_memory * 2)
                    batch_size.append(new_batch_size)
                else:
                    raise # some other error not memory related

            self.garbage_collection_cuda()
            if exit or count >= max_iters:
                break

        return batch_size[-1]


    def _compute_optimal_batch_size_predictive(self, method='train', max_memory=90,
                                               inputs=None, targets=None, pbar='n',
                                               max_iters=16, start_batch_size=4,
                                               delta_batch_size=4, max_batch_size=128,
                                               n=20, frequency=0.05, time_threshold=3,
                                               tail_size=20, std_threshold=0.1):
        """ Compute optimal batch size for training/inference to maximize GPU memory usage.
        Works by using `train`/`predict` with different batch sizes, and measuring how much memory is taken.
        Then, we solve the system of `measured_memory = batch_size * item_size + model_size + eps` equations for both
        `item_size` and `model_size`.

        For stable measurements, we make `n` iterations of `train`/`predict`, until the memory consumption stabilizes.
        """
        #pylint: disable=consider-iterating-dictionary
        table = {}
        batch_size = start_batch_size
        for _ in Notifier(pbar)(range(max_iters)):
            info = self.get_memory_utilization(batch_size, method=method,
                                               inputs=inputs, targets=targets, n=n, frequency=frequency,
                                               time_threshold=time_threshold,
                                               tail_size=tail_size, std_threshold=std_threshold)
            table[batch_size] = info

            # Exit condition
            batch_size += delta_batch_size
            if info['memory'] > max_memory or batch_size > max_batch_size:
                break

        # Make and solve a system of equations for `item_size`, `model_size`
        matrix = np.array([[batch_size, 1] for batch_size in table.keys()])
        vector = np.array([value['memory'] for value in table.values()])
        item_size, model_size = np.dot(np.linalg.pinv(matrix), vector)

        # Compute the `batch_size` to use up to `max_memory`
        optimal_batch_size = (max_memory - model_size) / item_size
        optimal_batch_size = int(optimal_batch_size)

        return {'batch_size': optimal_batch_size,
                'item_size': item_size,
                'model_size': model_size,
                'table': table}

    def get_memory_utilization(self, batch_size, method='train', inputs=None, targets=None, n=20, frequency=0.05,
                               time_threshold=3, tail_size=20, std_threshold=0.1):
        """ For a given `batch_size`, make `inputs` and `targets` and compute memory utilization. """
        inputs = inputs or self.make_placeholder_data(batch_size, to_device=False) # maybe use to_device=False
        inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
        inputs = [item[:batch_size] for item in inputs]

        targets = targets or self.predict(inputs=inputs, outputs='predictions')
        targets = list(targets) if isinstance(targets, (tuple, list)) else [targets]
        targets = [item[:batch_size] for item in targets]

        # Clear the GPU from potential previous runs
        self.garbage_collection_cuda()
        return self._get_memory_utilization(method=method, inputs=inputs, targets=targets, n=n, frequency=frequency,
                                            time_threshold=time_threshold,
                                            tail_size=tail_size, std_threshold=std_threshold)

    def _get_memory_utilization(self, method, inputs, targets, n, frequency,
                                time_threshold, tail_size, std_threshold):
        """ Run method `n` times and make sure that memory measurements are stable. """
        with GPUMemoryMonitor(frequency=frequency) as monitor:
            for _ in range(n):
                if method == 'train':
                    _ = self.train(inputs=inputs, targets=targets, microbatch_size=False)
                elif method == 'predict':
                    _ = self.predict(inputs=inputs, microbatch_size=False)

        if not self.config.get('benchmark'):
            data = monitor.data
            return {'memory': np.mean(data[-tail_size:]), 'n': n, 'monitor': monitor}

        # Check if the measurement is stable. If so, return the value and confidence
        data = monitor.data
        time = len(data) * frequency # in seconds
        if time > time_threshold:
            tail = data[-tail_size:]
            if np.std(tail) < std_threshold:
                return {'memory': np.mean(tail), 'n': n, 'monitor': monitor}

        # If the measurement is not stable, run for twice as long
        return self._get_memory_utilization(method=method, inputs=inputs, targets=targets,
                                            n=2*n, frequency=frequency, time_threshold=time_threshold,
                                            tail_size=tail_size, std_threshold=std_threshold)
