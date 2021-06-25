""" Tests for Research and correspong classes. """
# pylint: disable=no-name-in-module, missing-docstring, redefined-outer-name
import os
import pytest
from contextlib import ExitStack as does_not_raise

import numpy as np

from batchflow import Dataset, Pipeline, B, V, C
from batchflow import NumpySampler as NS
from batchflow.models.torch import ResNet
from batchflow.opensets import MNIST
from batchflow.research import Experiment, Executor, Domain, Option, Research, E, EC, O, ResearchResults

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
def complex_research():
    class Model:
        def __init__(self):
            self.dataset = MNIST()
            self.model_config = {
                'head/layout': C('layout'),
                'head/units': C('units'),
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
                .train_model('model', B('images'), B('labels'))
                .run_later(batch_size=8, n_iters=1, shuffle=True, drop_last=True)
            )
            self.train_ppl = ppl << self.dataset.train

        def create_test_ppl(self):
            test_ppl = (Pipeline()
                .import_model('model', self.train_ppl)
                .init_variable('metrics', None)
                .to_array(channels='first', src='images', dst='images')
                .predict_model('model', B('images'), fetches='predictions', save_to=B('predictions'))
                .gather_metrics('classification', B('labels'), B('predictions'), fmt='logits', axis=-1,
                                num_classes=10, save_to=V('metrics', mode='update'))
                .run_later(batch_size=8, n_iters=2, shuffle=False, drop_last=False)
            )
            self.test_ppl = test_ppl << self.dataset.test

        def eval_metrics(self, metrics, **kwargs):
            return self.test_ppl.v('metrics').evaluate(metrics, **kwargs)

    domain = Domain({'layout': ['f', 'faf']}) @ Domain({'units': [[10], [100, 10]]})
    research = (Research(domain=domain, n_reps=2)
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
            .save(E('ppl_branch').v('var'), dst='var', when=['last'])
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


class TestResearch:
    @pytest.mark.parametrize('parallel', [False, True])
    @pytest.mark.parametrize('dump_results', [False, True])
    @pytest.mark.parametrize('workers', [1, 3])
    @pytest.mark.parametrize('branches, target', [[1, 'f'], [3, 'f'], [3, 't']])
    def test_simple_research(self, parallel, dump_results, target, workers, branches, simple_research):
        n_iters = 3
        simple_research.run(n_iters=n_iters, workers=workers, branches=branches, parallel=parallel,
                            dump_results=dump_results, executor_target=target)

        assert len(simple_research.monitor.exceptions) == 0
        assert len(simple_research.results.df) == 18

    @pytest.mark.parametrize('parallel', [False, True])
    def test_load(self, parallel, simple_research):
        n_iters = 3
        simple_research.run(n_iters=n_iters, parallel=parallel, dump_results=True)

        loaded_research = Research.load(simple_research.name)

        assert len(loaded_research.results.df) == 18

    def test_empty_domain(self):
        research = Research().add_callable('func', lambda: 100).save(O('func'), 'sum')
        research.run(n_iters=10, dump_results=False)

        assert len(research.monitor.exceptions) == 0
        assert len(research.results.df) == 10

    def test_domain_update(self):
        def update():
            return Option('x', [4, 5, 6])

        research = (Research(domain=Option('x', [1, 2, 3]), n_reps=2)
            .add_callable('func', lambda x: x, x=EC('x'))
            .save(O('func'), 'sum')
            .update_domain(update, when=['%5', '%8'], n_reps=2)
        )
        research.run(n_iters=1, dump_results=False, bar=False)

        assert len(research.monitor.exceptions) == 0
        assert len(research.results.df) == 15

    @pytest.mark.slow
    @pytest.mark.parametrize('workers', [1, 2])
    def test_complex_research(self, workers, complex_research):
        complex_research.run(dump_results=False, parallel=True, workers=workers, bar=False, finalize=True)

        assert len(complex_research.monitor.exceptions) == 0
        assert len(complex_research.results.df) == 4

    def test_remove(self, simple_research):
        simple_research.run(n_iters=1)
        assert os.path.exists(simple_research.name)

        Research.remove(simple_research.name, ask=False)
        assert not os.path.exists(simple_research.name)

    @pytest.mark.parametrize('debug,expectation',
                             list(zip([False, True], [does_not_raise(), pytest.raises(NotImplementedError)])))
    def test_debug(self, debug, expectation):
        def func():
            raise NotImplementedError

        with expectation:
            research = Research().add_callable(func)
            research.run(dump_results=False, executor_target='f', parallel=False, debug=debug)


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


#TODO: logging tests, test that exceptions in one branch don't affect other bracnhes,
#      divices splitting, ...
