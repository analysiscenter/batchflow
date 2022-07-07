""" Tests for Research and correspong classes. """
# pylint: disable=no-name-in-module, missing-docstring, redefined-outer-name
import os
import sys
import glob
from contextlib import ExitStack as does_not_raise
import pytest
import psutil

import numpy as np

from batchflow import Dataset, Pipeline, B, V, C
from batchflow import NumpySampler as NS
from batchflow.models.torch import ResNet
from batchflow.opensets import CIFAR10
from batchflow.research import Experiment, Executor, Domain, Option, Research, E, EC, O, S, ResearchResults, Alias

class Model:
    def __init__(self):
        self.dataset = CIFAR10()
        self.model_config = {
            'head/layout': C('layout'),
            'head/features': C('features'),
            'classes': 10,
            'loss': 'ce',
            'device': 'cpu',
            'amp': False
        }
        self.create_train_ppl()
        self.create_test_ppl()

    def create_train_ppl(self):
        ppl = (Pipeline()
            .init_model('model', ResNet, 'dynamic', config=self.model_config)
            .to_array(channels='first', src='images', dst='images')
            .train_model('model', inputs=B('images'), targets=B('labels'))
            .run_later(batch_size=8, n_iters=1, shuffle=True, drop_last=True)
        )
        self.train_ppl = ppl << self.dataset.train

    def create_test_ppl(self):
        test_ppl = (Pipeline()
            .import_model('model', self.train_ppl)
            .init_variable('metrics', None)
            .to_array(channels='first', src='images', dst='images')
            .predict_model('model', inputs=B('images'), outputs='predictions', save_to=B('predictions'))
            .gather_metrics('classification', B('labels'), B('predictions'), fmt='logits', axis=-1,
                            num_classes=10, save_to=V('metrics', mode='update'))
            .run_later(batch_size=8, n_iters=2, shuffle=False, drop_last=False)
        )
        self.test_ppl = test_ppl << self.dataset.test

    def eval_metrics(self, metrics, **kwargs):
        return self.test_ppl.v('metrics').evaluate(metrics, **kwargs)

@pytest.fixture
def generator():
    def _generator(n):
        s = 0
        for i in range(n):
            s += i
            yield s
    return _generator

@pytest.fixture
def simple_research(tmp_path):
    def f(x, y):
        return x + y

    experiment = (Experiment()
        .add_callable('sum', f, x=EC('x'), y=EC('y'))
        .save(O('sum'), 'sum')
    )

    domain = Domain(x=[1, 2], y=[2, 3, 4])
    research = Research(name=os.path.join(tmp_path, 'research'), experiment=experiment, domain=domain)

    return research

@pytest.fixture
def research_with_controller(tmp_path):
    domain = Domain({'layout': ['f', 'faf']}) @ Domain({'features': [[10], [100, 10]]})
    research = (Research(name=os.path.join(tmp_path, 'research'), domain=domain, n_reps=2)
        .add_instance('controller', Model)
        .add_pipeline('controller.train_ppl')
        .add_pipeline('controller.test_ppl', run=True, when='last')
        .add_callable('controller.eval_metrics', metrics='accuracy', when='last')
        .save(O('controller.eval_metrics'), 'accuracy', when='last')
    )
    return research

SIZE_CALC = {
    '+': lambda x, y: x + y,
    '*': lambda x, y: x * y,
    '@': lambda x, y: x
}
class TestDomain:
    @pytest.mark.parametrize('op', ['+', '*', '@'])
    @pytest.mark.parametrize('a', [[0, 1, 2], [0, 1, 2, 4]])
    @pytest.mark.parametrize('b', [[2, 3, 4]])
    @pytest.mark.parametrize('n_reps', [1, 2])
    def test_operations(self, op, a, b, n_reps):
        option_1 = Domain({'a': a}) #pylint:disable=unused-variable
        option_2 = Domain(b=b) #pylint:disable=unused-variable

        if not (op == '@' and len(a) != len(b)):
            domain = eval(f'option_1 {op} option_2') # pylint:disable=eval-used
            domain.set_iter_params(n_reps=n_reps)

            configs = list(domain.iterator)
            n_items = SIZE_CALC[op](len(a), len(b))

            assert len(domain) == n_items
            assert domain.size == n_items * n_reps
            assert len(configs) == n_items * n_reps

    @pytest.mark.parametrize('repeat_each', [None, 1, 2])
    @pytest.mark.parametrize('n_reps', [1, 2, 3])
    def test_repetitions_order(self, repeat_each, n_reps):
        domain = Domain(a=[1, 2], b=[3, 4], c=NS('normal'))
        domain.set_iter_params(n_reps=n_reps, repeat_each=repeat_each)
        configs = list(domain.iterator)

        for i, config in enumerate(configs):
            if repeat_each is None:
                assert config.config()['repetition'] == i // len(domain)
            else:
                assert config.config()['repetition'] == i % (repeat_each * n_reps) // repeat_each

    def test_domain_update(self):
        domain = Domain({'a': [1, 2]})

        def update():
            return Domain({'x': [3, 4]})

        domain.set_update(update, ['last'])
        configs = list(domain.iterator)

        domain = domain.update(len(domain), None)
        configs += list(domain.iterator)

        assert len(configs) == 4
        for i, config in enumerate(configs):
            assert config.config()['updates'] == (2 * i) // len(configs)

    def test_sample_options(self):
        domain = Domain({'a': NS('normal')})
        domain.set_iter_params(n_items=3, n_reps=2, seed=42)

        res = [config['a'] for config in domain.iterator]
        exp_res = [0.03, 0.96, 0.73] * 2

        assert np.allclose(res, exp_res, atol=0.01, rtol=0)

    def test_weights(self):
        domain = (
            1. * Domain(a=[1,2]) +
            1. * Domain(b=[3,4]) +
            Domain(a=[5,6]) +
            0.3 * Domain(a=NS('normal')) +
            0.7 * Domain(b=NS('normal', loc=10))
        )
        domain.set_iter_params(n_items=8, n_reps=2, seed=41)
        res = [config['a'] if 'a' in config.config() else config['b'] for config in domain.iterator]
        exp_res = [1, 3, 4, 2, 5, 6, 9.87, -1.08] * 2

        assert np.allclose(res, exp_res, atol=0.01, rtol=0)

class TestExecutor:
    def test_callable(self):
        experiment = (Experiment()
            .add_callable('sum', sum, args=[range(10)])
            .save(O('sum'), 'sum')
        )
        executor = Executor(experiment, target='f', n_iters=1)
        executor.run()

        assert executor.experiments[0].results['sum'][0] == sum(range(10))

    def test_generator(self, generator):
        experiment = (Experiment()
            .add_generator('sum', generator, n=10)
            .save(O('sum'), 'sum')
        )

        executor = Executor(experiment, target='f', n_iters=10)
        executor.run()

        assert executor.experiments[0].results['sum'][9] == sum(range(10))

    def test_direct_callable(self):
        experiment = (Experiment()
            .sum(range(10), save_to='sum')
        )
        executor = Executor(experiment, target='f', n_iters=1)
        executor.run()

        assert executor.experiments[0].results['sum'][0] == sum(range(10))

    def test_direct_generator(self, generator): #pylint: disable=unused-argument
        experiment = (Experiment()
            .add_namespace(locals())
            .generator(10, mode='generator')
            .save(O('generator'), 'sum')
        )

        executor = Executor(experiment, target='f', n_iters=10)
        executor.run()

        assert executor.experiments[0].results['sum'][9] == sum(range(10))

    def test_units_without_name(self, generator):
        experiment = (Experiment()
            .add_callable(sum, args=[range(10)])
            .add_generator(generator, n=10)
            .save(O('sum'), 'sum')
        )
        executor = Executor(experiment, target='f', n_iters=1)
        executor.run()

        assert executor.experiments[0].results['sum'][0] == sum(range(10))

    def test_configs(self):
        def f(x, y, z):
            return (x, y, z)

        experiment = (Experiment()
            .add_callable('sum', f, x=EC('x'), y=EC('y'), z=EC('z'), save_to='sum')
        )

        executor = Executor(experiment, target='f', configs=[{'x': 10}, {'x': 20}],
                            branches_configs=[{'y': 20}, {'y': 30}], executor_config={'z': 5},
                            n_iters=1)
        executor.run()

        assert executor.experiments[0].results['sum'][0] == (10, 20, 5)
        assert executor.experiments[1].results['sum'][0] == (20, 30, 5)

    def test_root(self):
        def root():
            return 10

        experiment = (Experiment()
            .add_callable('root', root, root=True)
            .add_callable('sum', sum, args=[[EC('x'), O('root')]])
            .save(E().outputs['sum'], 'sum')
        )

        executor = Executor(experiment, target='f', configs=[{'x': 10}, {'x': 20}], n_iters=1)
        executor.run()

        assert executor.experiments[0].results['sum'][0] == 20
        assert executor.experiments[1].results['sum'][0] == 30

    def test_instances(self):
        class MyClass:
            def __init__(self, x):
                self.x = x

            def sum(self):
                return sum(range(self.x))

        experiment = (Experiment()
            .add_instance('instance', MyClass, x=EC('x'))
            .add_callable('instance.sum')
            .save(O('instance.sum'), 'sum')
        )

        executor = Executor(experiment, target='f', configs=[{'x': 10}, {'x': 20}], n_iters=1)
        executor.run()

        assert executor.experiments[0].results['sum'][0] == sum(range(10))
        assert executor.experiments[1].results['sum'][0] == sum(range(20))

    def test_pipeline(self):
        ppl = (Dataset(10).p
            .init_variable('var', 0)
            .update(V('var'), V('var') + B().indices.sum())
            .run_later(1, n_epochs=1, shuffle=False)
        )

        experiment = (Experiment()
            .add_pipeline('ppl', ppl)
            .save(E('ppl').v('var'), dst='var', when=['last'])
        )

        executor = Executor(experiment, target='f', n_iters=10)
        executor.run()

        assert executor.experiments[0].results['var'][9] == sum(range(10))

    def test_pipeline_with_branches(self):
        root = Dataset(10).p.run_later(1, n_epochs=1, shuffle=False)
        ppl = (Pipeline()
            .init_variable('var', 0)
            .update(V('var'), V('var') + B().indices.sum() * C('x'))
        )

        experiment = (Experiment()
            .add_pipeline('ppl', root, ppl)
            .save(E('ppl').v('var'), dst='var', when=['last'])
        )

        executor = Executor(experiment, target='f', n_iters=10, configs=[{'x': 10}, {'x': 20}], )
        executor.run()

        assert executor.experiments[0].results['var'][9] == sum(range(10)) * 10
        assert executor.experiments[1].results['var'][9] == sum(range(10)) * 20

    def test_stop_iteration(self, generator):
        def inc(x):
            return x + 1

        experiment = (Experiment()
            .add_generator('sum', generator, n=EC('n'))
            .add_callable('func', inc, x=O('sum'))
            .save(O('sum'), 'sum', when='last')
            .save(O('func'), 'func', when='last')
        )

        executor = Executor(experiment, target='f', configs=[{'n': 10}, {'n': 20}], n_iters=30, finalize=True)
        executor.run()

        assert executor.experiments[0].results['sum'][10] == sum(range(10))
        assert executor.experiments[1].results['sum'][20] == sum(range(20))

        assert executor.experiments[0].results['func'][10] == sum(range(10)) + 1
        assert executor.experiments[1].results['func'][20] == sum(range(20)) + 1

        executor = Executor(experiment, target='f', configs=[{'n': 10}, {'n': 20}], n_iters=None)
        executor.run()

    @pytest.mark.parametrize('save_to, save_output_dict, expectation', [
        ['a', False, does_not_raise()],
        [['a'], False, pytest.raises(ValueError)],
        [['a', 'b'], False, pytest.raises(ValueError)],
        [['a', 'b', 'c'], False, does_not_raise()],
        [None, True, does_not_raise()]
    ])
    def test_multiple_output(self, save_to, save_output_dict, expectation):
        def func():
            return {'a': 1, 'b': 2, 'c': 3}

        experiment = Experiment().add_callable(func, save_to=save_to, save_output_dict=save_output_dict)
        executor = Executor(experiment, target='f', n_iters=1, debug=True)

        with expectation:
            executor.run()

        executor.close()

        process = psutil.Process(os.getpid())
        assert len(process.children()) <= 1

class TestResearch:
    @pytest.mark.parametrize('parallel', [False, True])
    @pytest.mark.parametrize('dump_results', [False, True])
    @pytest.mark.parametrize('workers', [1, 3])
    @pytest.mark.parametrize('branches, target', [[1, 'f'], [3, 'f'], [3, 't']])
    def test_simple_research(self, parallel, dump_results, target, workers, branches, simple_research):
        n_iters = 3

        process = psutil.Process(os.getpid())
        assert len(process.children()) <= 1
        simple_research.run(n_iters=n_iters, workers=workers, branches=branches, parallel=parallel,
                            dump_results=dump_results, executor_target=target)

        assert len(simple_research.monitor.exceptions) == 0
        assert len(simple_research.results.df) == 18

        if dump_results:
            loaded_research = Research.load(simple_research.name)
            assert len(loaded_research.results.df) == 18

        process = psutil.Process(os.getpid())
        assert len(process.children()) <= 1

    def test_empty_domain(self):
        research = Research().add_callable('func', lambda: 100).save(O('func'), 'sum')
        research.run(n_iters=10, dump_results=False)

        assert len(research.monitor.exceptions) == 0
        assert len(research.results.df) == 10

    @pytest.mark.parametrize('create_id_prefix', [False, True])
    def test_domain_update(self, create_id_prefix):
        def update():
            return Option('x', [4, 5, 6])

        research = (Research(domain=Option('x', [1, 2, 3]), n_reps=2)
            .add_callable('func', lambda x: x, x=EC('x'))
            .save(O('func'), 'sum')
            .update_domain(update, when=['%5', '%8'], n_reps=2)
        )
        research.run(n_iters=1, dump_results=False, bar=False, create_id_prefix=create_id_prefix)

        assert len(research.monitor.exceptions) == 0
        assert len(research.results.df) == 15

    @pytest.mark.slow
    @pytest.mark.parametrize('workers', [1, 2])
    def test_research_with_controller(self, workers, research_with_controller):
        research_with_controller.run(dump_results=True, parallel=True, workers=workers, bar=False, finalize=True)

        process = psutil.Process(os.getpid())

        assert len(research_with_controller.monitor.exceptions) == 0
        assert len(research_with_controller.results.df) == 4
        assert len(process.children()) <= 1

        loaded_research = Research.load(research_with_controller.name)
        assert len(loaded_research.results.df) == 4

        assert len(process.children()) <= 1

    @pytest.mark.slow
    @pytest.mark.parametrize('branches', [False, True])
    def test_research_with_pipelines(self, branches):
        dataset = CIFAR10()
        model_config = {
            'head/layout': C('layout'),
            'head/features': C('features'),
            'loss': 'ce',
            'classes': 10,
            'device': 'cpu',
            'amp': False
        }

        root_ppl = (Pipeline()
            .to_array(channels='first', src='images', dst='images')
            .run_later(batch_size=8, n_iters=1, shuffle=True, drop_last=True)
        ) << dataset.train

        branch_ppl = (Pipeline()
            .init_model('model', ResNet, 'dynamic', config=model_config)
            .init_variable('loss', None)
            .train_model('model', inputs=B('images'), targets=B('labels'), outputs='loss', save_to=V('loss'))
        )

        test_ppl = (Pipeline()
            .import_model('model', C('import_from'))
            .init_variable('metrics', None)
            .to_array(channels='first', src='images', dst='images')
            .predict_model('model', inputs=B('images'), outputs='predictions', save_to=B('predictions'))
            .gather_metrics('classification', B('labels'), B('predictions'), fmt='logits', axis=-1,
                            num_classes=10, save_to=V('metrics', mode='update'))
            .run_later(batch_size=8, n_iters=2, shuffle=False, drop_last=False)
        ) << dataset.test

        def eval_metrics(ppl, metrics, **kwargs):
            return ppl.v('metrics').evaluate(metrics, **kwargs)

        domain = Domain({'layout': ['f', 'faf']}) @ Domain({'features': [[10], [100, 10]]})

        args = (root_ppl, branch_ppl) if branches else (root_ppl+branch_ppl, )

        research = (Research(name='research', domain=domain, n_reps=2)
            .add_pipeline('train_ppl', *args, variables='loss')
            .add_pipeline('test_ppl', test_ppl, import_from=E('train_ppl'), run=True, when='last')
            .add_callable(eval_metrics, ppl=E('test_ppl'), metrics='accuracy', when='last')
            .save(O('eval_metrics'), 'accuracy', when='last')
        )

        research.run(dump_results=False)

        results = research.results.df.dtypes.values

        # columns : id, layout, units, iteration, loss, accuracy
        assert all(results == [np.dtype(i) for i in ['O', 'O', 'O', 'int64', 'float32', 'float64']])

        process = psutil.Process(os.getpid())
        for p in process.children():
            print(p, research.monitor.processes.get(p.pid, 'unknown'))
        assert len(process.children()) <= 1

    def test_update_variable(self):
        def my_call(x, storage):
            if x > 2:
                storage.update_variable('var1', x ** 2)
            else:
                storage.update_variable('var2', x ** 2)
        research = (Research(domain=Option('x', [1, 2, 3, 4]))
            .add_callable('func', my_call, x=EC('x'), storage=S())
        )

        research.run(workers=2, branches=2, dump_results=False, bar=False)

        results = research.results.df.sort_values('x')
        var1 = results['var1'].values
        var2 = results['var2'].values
        a = np.array([np.nan, np.nan, 9., 16.])
        b = np.array([1., 4., np.nan, np.nan])
        assert ((var1 == a) | (np.isnan(var1) & np.isnan(a))).all() and \
               ((var2 == b) | (np.isnan(var2) & np.isnan(b))).all()

    @pytest.mark.parametrize('create_id_prefix', [False, True, 4])
    @pytest.mark.parametrize('domain', [
        Domain(x=[1, 2], y=[2, 3, 4]),
        Domain(x=[1, 2]) @ Domain(y=[2, 3]),
        None
    ])
    def test_prefixes(self, tmp_path, create_id_prefix, domain):
        research = (
            Research(name=os.path.join(tmp_path, 'research'), domain=domain)
            .add_callable(lambda: 1, save_to='a')
        )
        research.run(dump_results=True, n_iters=1, create_id_prefix=create_id_prefix)

        if create_id_prefix is False:
            # id includes only hash
            assert research.results.df.id.apply(lambda x: len(x.split('_')) == 1).all()
        else:
            # id includes prefix for each parameter, repetition index and hash
            n_params = 2 if domain is not None else 0
            assert research.results.df.id.apply(lambda x: len(x.split('_')) == n_params + 2).all()
            parsed_id = research.results.df.id.apply(lambda x: x.split('_'))

            # check the number of digits for each prefix code
            assert parsed_id.apply(lambda x: all([len(i) == create_id_prefix for i in x[:-1]])).all()


    def test_remove(self, simple_research):
        simple_research.run(n_iters=1)
        assert os.path.exists(simple_research.name)

        Research.remove(simple_research.name, ask=False)
        assert not os.path.exists(simple_research.name)

    @pytest.mark.slow
    @pytest.mark.parametrize('debug,expectation',
                             list(zip([False, True], [does_not_raise(), pytest.raises(NotImplementedError)])))
    def test_debug(self, debug, expectation):
        def func():
            raise NotImplementedError

        with expectation:
            research = Research().add_callable(func)
            research.run(dump_results=False, executor_target='f', parallel=False, debug=debug)

    @pytest.mark.parametrize('profile, shape', list(zip([2, 1], [9, 6])))
    def test_profile(self, profile, shape, simple_research):
        simple_research.run(n_iters=3, dump_results=False, profile=profile)

        assert simple_research.profiler.profile_info.shape[1] == shape

    @pytest.mark.parametrize('loglevel, length_res, length_exp', list(zip(['info', 'debug'], [62, 104], [7, 14])))
    def test_logging(self, loglevel, length_res, length_exp, tmp_path, simple_research):
        path = os.path.join(tmp_path, 'research')
        simple_research.run(name=path, n_iters=3, dump_results=True, loglevel=loglevel)

        with open(os.path.join(path, 'research.log')) as file:
            lines = file.readlines()
            assert len(lines) == length_res

        for path in glob.glob(os.path.join(path, 'experiments', '*')):
            with open(os.path.join(path, 'experiment.log')) as file:
                lines = file.readlines()
                assert len(lines) == length_exp

    def test_coincided_names(self):
        def f(a):
            return a ** 10

        research = (Research()
            .add_callable(f, a=2, save_to='a')
            .add_callable(f, a=3, save_to='b')
        )

        research.run(dump_results=False)

        assert research.results.df.iloc[0].a == f(2)
        assert research.results.df.iloc[0].b == f(3)

    @pytest.mark.parametrize('dump_results', [False, True])
    @pytest.mark.parametrize('redirect_stdout', [True, 0, 1, 2, 3])
    @pytest.mark.parametrize('redirect_stderr', [True, 0, 1, 2, 3])
    def test_redirect_stdout(self, dump_results, redirect_stdout, redirect_stderr, tmp_path):
        def f(a):
            print(a)
            print(a, file=sys.stderr)

        research = (Research()
            .add_callable(f, a=2)
        )

        will_success = False

        if dump_results:
            will_success = True
        else:
            if not dump_results:
                if redirect_stdout in [0, 2] or redirect_stdout is True:
                    if redirect_stderr in [0, 2] or redirect_stderr is True:
                        will_success = True
        expectation = does_not_raise() if will_success else pytest.raises(ValueError)

        with expectation:
            research.run(name=os.path.join(tmp_path, 'research'), n_iters=2, dump_results=dump_results,
                        redirect_stdout=redirect_stdout, redirect_stderr=redirect_stderr)

        if will_success:
            for param, name in zip([redirect_stdout, redirect_stderr], ['stdout', 'stderr']):
                output = '2\n' + '-' * 30 + '\n\n\n' + '-'*30 + '\n'
                filename = name + '.txt'

                if dump_results:
                    n_files = len(research.results.artifacts_to_df(name=filename))
                    assert (filename in os.listdir(research.name)) is (param in [1, 3])
                    assert (n_files == len(research.results.configs)) is (param in [2, 3])

                if param in [True, 1, 3]:
                    if dump_results:
                        with open(os.path.join(research.name, filename)) as file:
                            lines = ''.join(file.readlines())
                    else:
                        lines = list(getattr(research.storage, 'experiments_'+name).values())[0]
                    assert lines == output * 2

                if dump_results and param in [2, 3]:
                    for full_path in research.results.artifacts_to_df(name=filename)['full_path']:
                        with open(full_path) as file:
                            lines = ''.join(file.readlines())
                            assert lines == output * 2

    def test_domain_with_objects(self, tmp_path):
        domain = Domain(a=[print, Research])
        research = Research(domain=domain).add_callable(lambda: 1)
        research.run(name=os.path.join(tmp_path, 'research'), parallel=False)

class TestResults:
    @pytest.mark.parametrize('parallel', [False, True])
    @pytest.mark.parametrize('dump_results', [False, True])
    def test_filter_by_config(self, parallel, dump_results, simple_research):
        simple_research.run(n_iters=3, parallel=parallel, dump_results=dump_results)
        df = simple_research.results.to_df(use_alias=False, config={'y': 2})

        assert len(df) == 6
        assert (df.y.values == 2).all()

    @pytest.mark.parametrize('parallel', [False, True])
    @pytest.mark.parametrize('dump_results', [False, True])
    def test_filter_by_alias(self, parallel, dump_results, simple_research):
        simple_research.run(n_iters=3, parallel=parallel, dump_results=dump_results)
        df = simple_research.results.to_df(use_alias=False, alias={'y': '2'})

        assert len(df) == 6
        assert (df.y.values == 2).all()

    @pytest.mark.parametrize('parallel', [False, True])
    @pytest.mark.parametrize('dump_results', [False, True])
    def test_filter_by_domain(self, parallel, dump_results, simple_research):
        simple_research.run(n_iters=3, parallel=parallel, dump_results=dump_results)
        df = simple_research.results.to_df(use_alias=False, domain=Option('y', [2, 3]))

        assert len(df) == 12

    def test_load(self, simple_research):
        simple_research.run(n_iters=3)

        df = ResearchResults(simple_research.name, domain=Option('y', [2, 3])).df
        assert len(df) == 12

    @pytest.mark.parametrize('use_alias', [False, True])
    def test_lists_in_config(self, use_alias):
        def dummy():
            return 1

        domain = Domain({'a': [Alias([1, 2, 3], 'list')]})
        research = Research(domain=domain).add_callable(dummy, save_to='res')
        research.run(dump_results=False)

        df = research.results.to_df(use_alias=use_alias)

        assert len(df) == 1

# #TODO: test that exceptions in one branch don't affect other bracnhes,
# #      devices splitting, ...
