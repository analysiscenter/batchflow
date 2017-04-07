""" Pipeline classes and decorators """


def action(method):
    """ Decorator for action methods in Batch classes """
    # TODO: decorator params: parallelization (e.g. threads, processes, async/await, greenlets,...)
    # use __action for class-specific params
    method.action = True
    return method


class Pipeline:
    """ Pipeline """
    def __init__(self, dataset):
        self.dataset = dataset
        self.action_list = []
        self.batch_generator = None


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
            raise AttributeError("Method %s has not been found in Preprocessing and Batch classes" % name)
        return self._append_action


    @property
    def index(self):
        """ Return index of the source dataset """
        return self.dataset.index

    def __len__(self):
        """ Return index length """
        return len(self.index)


    def _append_action(self, *args, **kwargs):
        """ Add new action to the log of future actions """
        self.action_list[-1].update({'args': args, 'kwargs': kwargs})
        return self


    def _exec_all_actions(self, batch):
        batch_res = self
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
                batch_res = batch_action(*_action_args, **_action['kwargs'])
        return batch_res

    def join(self, datasets):
        """ Join other datasets """
        self.action_list.append({'name': 'join', 'datasets': datasets})
        return self

    def _run_seq(self, gen_batch):
        for batch in gen_batch:
            batch_res = self._exec_all_actions(batch)
        return batch_res


    def run(self, batch_size, shuffle=False, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset
            Batches are created sequentially, one after another, without batch-level parallelism
        """
        batch_generator = self.dataset.gen_batch(batch_size, shuffle=shuffle, one_pass=True, *args, **kwargs)
        self._run_seq(batch_generator)
        return self


    def create_batch(self, batch_indices, *args, **kwargs):
        """ Create a new batch by given indices and execute all previous lazy actions """
        batch = self.dataset.create_batch(batch_indices, *args, **kwargs)
        batch_res = self._exec_all_actions(batch)
        return batch_res


    def next_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Get the next batch and execute all previous lazy actions """
        if self.batch_generator is None:
            self.batch_generator = self.dataset.gen_batch(batch_size, shuffle=shuffle,
                                                          one_pass=one_pass, *args, **kwargs)
        batch = next(self.batch_generator)
        batch_res = self._exec_all_actions(batch)
        return batch_res
