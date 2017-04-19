""" Pipeline classes """
import concurrent.futures as cf
import asyncio
import queue as q


class Pipeline:
    """ Pipeline """
    def __init__(self, dataset):
        self.dataset = dataset
        self.action_list = []
        self.prefetch_queue = None
        self.executor = None

    def __getattr__(self, name, *args, **kwargs):
        """ Check if an unknown attr is an action from the batch class """
        if hasattr(self.dataset.batch_class, name):
            attr_name = getattr(self.dataset.batch_class, name)
            if callable(attr_name):
                if hasattr(attr_name, "action"):
                    self.action_list.append({'name': name})
                else:
                    raise ValueError("Method %s is not marked with @action decorator" % name)
        else:
            raise AttributeError("Method %s has not been found in Pipeline and Batch classes" % name)
        return self._append_action

    def __getstate__(self):
        return {'dataset': self.dataset, 'action_list': self.action_list}

    def __setstate__(self, state):
        self.dataset = state['dataset']
        self.action_list = state['action_list']

    @property
    def index(self):
        """ Return index of the source dataset """
        return self.dataset.index

    @property
    def indices(self):
        """ Return the sequence of indices of the source dataset """
        return self.index.indices

    def __len__(self):
        """ Return index length """
        return len(self.index)

    def _append_action(self, *args, **kwargs):
        """ Add new action to the log of future actions """
        self.action_list[-1].update({'args': args, 'kwargs': kwargs})
        return self

    def _exec_all_actions(self, batch, new_loop=False):
        if new_loop:
            asyncio.set_event_loop(asyncio.new_event_loop())

        joined_sets = None
        for _action in self.action_list:
            if _action['name'] == 'join':
                joined_sets = _action['datasets']
            else:
                batch_action = getattr(batch, _action['name'])
                if joined_sets is not None:
                    joined_data = []
                    if not isinstance(joined_sets, (list, tuple)):
                        joined_sets = [joined_sets]
                    for jset in joined_sets:
                        joined_data.append(jset.create_batch(batch.index))
                    _action_args = (joined_data,) + _action['args']
                    joined_sets = None
                else:
                    _action_args = _action['args']
                batch = batch_action(*_action_args, **_action['kwargs'])
        return batch

    def join(self, datasets):
        """ Join other datasets """
        self.action_list.append({'name': 'join', 'datasets': datasets})
        return self

    def _run_seq(self, gen_batch):
        for batch in gen_batch:
            _ = self._exec_all_actions(batch)

    def _put_batches_into_queue(self, gen_batch):
        for batch in gen_batch:
            future = self.executor.submit(self._exec_all_actions, batch, True)
            self.prefetch_queue.put(future, block=True)
        self.prefetch_queue.put(None, block=True)

    def _run_batches_from_queue(self):
        while True:
            future = self.prefetch_queue.get(block=True)
            if future is None:
                self.prefetch_queue.task_done()
                break
            else:
                _ = future.result()
                self.prefetch_queue.task_done()
        return None

    def run(self, batch_size, shuffle=False, one_pass=True, prefetch=0, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset
            Batches are created sequentially, one after another, without batch-level parallelism
        """
        if 'target' in kwargs:
            target = kwargs['target']
            del kwargs['target']
        else:
            target = 'threads'

        batch_generator = self.dataset.gen_batch(batch_size, shuffle, one_pass, *args, **kwargs)

        if prefetch > 0:
            if target == 'threads':
                self.executor = cf.ThreadPoolExecutor(max_workers=prefetch)
            elif target == 'mpc':
                self.executor = cf.ProcessPoolExecutor(max_workers=prefetch)   # pylint: disable=redefined-variable-type
            else:
                raise ValueError("target should be one of ['threads', 'mpc']")

            self.prefetch_queue = q.Queue(maxsize=prefetch)
            service_executor = cf.ThreadPoolExecutor(max_workers=2)
            service_executor.submit(self._put_batches_into_queue, batch_generator)
            future = service_executor.submit(self._run_batches_from_queue)
            # wait until all batches have been processed
            _ = future.result()
        else:
            self.prefetch_queue = None
            self.executor = None
            self._run_seq(batch_generator)
        return self

    def create_batch(self, batch_index, *args, **kwargs):
        """ Create a new batch by given indices and execute all previous lazy actions """
        batch = self.dataset.create_batch(batch_index, *args, **kwargs)
        batch_res = self._exec_all_actions(batch)
        return batch_res

    def next_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Get the next batch and execute all previous lazy actions """
        batch_index = self.index.next_batch(batch_size, shuffle, one_pass, *args, **kwargs)
        batch_res = self.create_batch(batch_index, *args, **kwargs)
        return batch_res
