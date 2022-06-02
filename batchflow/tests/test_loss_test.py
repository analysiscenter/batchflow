""" Test for test loss as metric """
# pylint: disable=import-error, no-name-in-module

import numpy as np
import pytest

from batchflow import Dataset, B, V
from batchflow.research import Research, E, get_metrics


np.random.seed(2020)
NUM_ITEMS = 10
DATA = np.random.sample(NUM_ITEMS)


@pytest.mark.parametrize('batch_size', [4, 5])
class TestTestLoss:
    """
    Test loss as Metrics
    """
    @staticmethod
    @pytest.fixture()
    def research_path(tmp_path):
        """
        Make path in temporary pytest folder for research
        """
        return str((tmp_path / 'tmp').absolute())

    def test_research(self, batch_size, research_path):
        """ test acquisition of test loss in research with pipeline added with `run=True`"""
        dataset = Dataset(index=NUM_ITEMS, preloaded=DATA)

        ppl = (dataset.p
               .init_variable("metric_test_loss")
               .gather_metrics('loss', loss=B.data.mean(), batch_len=B.data.shape[0],
                               save_to=V('metric_test_loss', mode='a'))
               .run_later(batch_size, n_epochs=1)
               )

        research = (Research()
                    .add_pipeline('ppl', ppl, when='last', run=True)
                    .add_callable(get_metrics, pipeline=E('ppl').pipeline, metrics_var='metric_test_loss',
                                  metrics_name='loss', save_to='test_loss', when='last')
                    )

        research.run(name=research_path, n_iters=1)

        assert np.allclose(DATA.mean(), research.results.df.test_loss[0])
