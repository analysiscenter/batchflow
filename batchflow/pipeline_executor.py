""" Contains pipeline executor class """
import warnings
import traceback
import concurrent.futures as cf
import queue as q

from .exceptions import SkipBatchException, EmptyBatchSequence, StopPipeline
from .notifier import Notifier

warnings.filterwarnings("always", category=RuntimeWarning, module=__name__)
warnings.filterwarnings("always", category=EmptyBatchSequence)


class PipelineExecutor:
    """ Pipeline executor"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.notifier = None

        self._stop_flag = False
        self._executor = None
        self._service_executor = None
        self._prefetch_count = None
        self._prefetch_queue = None
        self._batch_queue = None

    def _clear_queue(self, queue):
        if queue is not None:
            while not queue.empty():
                queue.get(block=True)
                queue.task_done()

    def _stop_executor(self, executor):
        if executor is not None:
            executor.shutdown()

    def reset(self):
        """ Clear all iteration metadata in order to start iterating from scratch """
        self._stop_flag = True

        self._clear_queue(self._prefetch_count)
        self._clear_queue(self._prefetch_queue)
        self._clear_queue(self._batch_queue)

        self._stop_executor(self._executor)
        self._stop_executor(self._service_executor)

        self._executor = None
        self._service_executor = None
        self._prefetch_count = None
        self._prefetch_queue = None
        self._batch_queue = None


    def _put_batches_into_queue(self, gen_batch):
        iteration = 1
        while not self._stop_flag:
            # this will block if there are too many batches are prefetched
            self._prefetch_count.put(1, block=True)
            try:
                batch = next(gen_batch)
            except StopIteration:
                break
            else:
                future = self._executor.submit(self.pipeline.execute_for, batch, iteration=iteration, new_loop=True)
                self._prefetch_queue.put(future, block=True)
                iteration = iteration + 1
        self._prefetch_queue.put(None, block=True)

    def _run_batches_from_queue(self, notifier):
        while not self._stop_flag:
            future = self._prefetch_queue.get(block=True)
            if future is None:
                self._prefetch_queue.task_done()
                self._batch_queue.put(None)
                break

            try:
                batch = future.result()
            except SkipBatchException:
                pass
            except StopPipeline:
                self._batch_queue.put(None)
                break
            except Exception:   # pylint: disable=broad-except
                exc = future.exception()
                print("Exception in a thread:", exc)
                traceback.print_tb(exc.__traceback__)
                self._prefetch_count.get(block=True)
                self._prefetch_count.task_done()
            else:
                notifier.update(pipeline=self, batch=batch)
                self._batch_queue.put(batch, block=True)
            finally:
                self._prefetch_queue.task_done()


    def gen_batch(self, *args, dataset=None, rebatch=False, reset='iter', profile=False, **kwargs):
        """ Generate batches

        Parameters
        ----------
        batch_size : int
            desired number of items in the batch (the actual batch could contain fewer items)

        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            specifies the order of items, could be:

            - bool - if `False`, items go sequentionally, one after another as they appear in the index.
                if `True`, items are shuffled randomly before each epoch.

            - int - a seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance.

            - callable - a function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

        n_iters : int
            Number of iterations to make (only one of `n_iters` and `n_epochs` should be specified).

        n_epochs : int
            Number of epochs required (only one of `n_iters` and `n_epochs` should be specified).

        drop_last : bool
            if `True`, drops the last batch (in each epoch) if it contains fewer than `batch_size` items.

            If `False`, than the last batch in each epoch could contain repeating indices (which might be a problem)
            and the very last batch could contain fewer than `batch_size` items.

            See :meth:`~.DatasetIndex.gen_batch` for details.

        notifier : str, dict, or instance of :class:`~.Notifier`
            Configuration of displayed progress notifiers (like bar, etc), if any.
            If str or dict, then parameters of :class:`~.Notifier` initialization.

        prefetch : int
            a number of batches to process in advance (default=0)

        target : 'threads' or 'mpc'
            batch parallelization engine used for prefetching (default='threads').
            'mpc' rarely works well due to complicated and slow python's inter-process communications.

        reset : list of str, str or bool
            what to reset to start from scratch:

            - 'iter' - restart the batch iterator
            - 'variables' - re-initialize all pipeline variables
            - 'models' - reset all models

        dataset
            a dataset to get batches from

        rebatch : bool
            if rebatching is needed

        Yields
        ------
        an instance of the batch class returned by the last action

        Examples
        --------

        ::

            for batch in pipeline.gen_batch(C('batch_size'), shuffle=True, n_epochs=2, drop_last=True):
                # do whatever you want
        """
        self.reset()
        self.pipeline.reset(reset, profile=profile)

        if 'n_iters' not in kwargs and 'n_epochs' not in kwargs:
            kwargs.setdefault('n_epochs', 1)
        target = kwargs.pop('target', 'threads')
        prefetch = kwargs.pop('prefetch', 0)

        if 'bar' in kwargs:
            warnings.warn('`bar` argument is deprecated and renamed to `notifier`', DeprecationWarning, stacklevel=2)
        notifier = kwargs.pop('notifier', kwargs.pop('bar', None))

        if rebatch:
            batch_generator = self.pipeline.gen_rebatch(*args, prefetch=prefetch, **kwargs)
            prefetch = 0
        else:
            batch_generator = dataset.gen_batch(*args, iter_params=self.pipeline.iter_params, **kwargs)

        batch_size = args[0] if len(args) != 0 else kwargs.get('batch_size')
        n_iters = kwargs.get('n_iters')
        n_epochs = kwargs.get('n_epochs')
        drop_last = kwargs.get('drop_last')

        if not isinstance(notifier, Notifier):
            notifier = Notifier(**(notifier if isinstance(notifier, dict) else {'bar': notifier}),
                                batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                                drop_last=drop_last, length=len(self.pipeline))
        if notifier.total is None:
            notifier.update_total(total=None, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                                  drop_last=drop_last, length=len(self.pipeline))

        if self.pipeline.before:
            self.pipeline.before.run()

        if prefetch > 0:
            if target in ['threads', 't']:
                self._executor = cf.ThreadPoolExecutor(max_workers=prefetch + 1)
            elif target in ['mpc', 'm']:
                self._executor = cf.ProcessPoolExecutor(max_workers=prefetch + 1)
            else:
                raise ValueError("target should be one of ['threads', 'mpc']")

            self._stop_flag = False
            # count queue warrants that exactly prefetch+1 num batches will be submitted to executor
            # it serves as a gate
            self._prefetch_count = q.Queue(maxsize=prefetch+1)
            # prefetch queue holds futures of batches being processed now
            # most of the time exactly prefetch+1 items will be in the queue
            self._prefetch_queue = q.Queue()
            # batch queue holds batches ready to be yielded
            self._batch_queue = q.Queue()
            # due to count queue both prefetch and batch queue cannot contain more than prefetch+1 items

            # service executor runs batch generation and batch processing threads
            self._service_executor = cf.ThreadPoolExecutor(max_workers=2)
            # this thread submits batches (waits for count queue and puts into prefetch queue)
            self._service_executor.submit(self._put_batches_into_queue, batch_generator)
            # this thread gets processed batches (waits for futures to be complete and puts into batch queue)
            self._service_executor.submit(self._run_batches_from_queue, notifier)

            # main thread gets ready batches and yield them one by one (releasing count queue after each one)
            while not self._stop_flag:
                batch_res = self._batch_queue.get(block=True)
                self._batch_queue.task_done()
                if batch_res is None:
                    self._stop_flag = True
                else:
                    # the batch has been created in another thread, so we need to set pipeline
                    batch_res.pipeline = self.pipeline
                    yield batch_res
                    self._prefetch_count.get(block=True)
                    self._prefetch_count.task_done()

        else:
            is_empty = True
            iteration = 1
            for batch in batch_generator:
                try:
                    batch_res = self.pipeline.execute_for(batch, notifier, iteration)
                except SkipBatchException:
                    pass
                except StopPipeline:
                    break
                else:
                    is_empty = False
                    yield batch_res
                    iteration = iteration + 1
            if is_empty:
                warnings.warn("Batch generator is empty. Use pipeline.reset('iter') to restart iteration.",
                              EmptyBatchSequence, stacklevel=3)

        notifier.close()
        self.notifier = notifier

        if self.pipeline.after:
            self.pipeline.after.run()


    def run(self, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset

        See also
        --------
        :meth:`~.PipelineExecutor.gen_batch`
        """
        for _ in self.pipeline.gen_batch(*args, **kwargs):
            pass

        return self.pipeline
