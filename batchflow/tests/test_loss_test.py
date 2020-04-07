""" Test for test loss as metric """
# pylint: disable=import-error, no-name-in-module

import numpy as np
import pytest

from batchflow import Dataset, B, V
from batchflow.research import Research


np.random.seed(2020)
NUM_ITEMS = 10
DATA = np.random.sample(NUM_ITEMS)


@pytest.mark.skip(reason="check pytest failure in github actions")
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
                    .add_pipeline(ppl, name='ppl', execute='last', run=True)
                    .get_metrics(pipeline='ppl', metrics_var='metric_test_loss', metrics_name='loss',
                                 returns='test_loss', execute='last')
                    )

        research.run(1, name=research_path)

        assert np.allclose(DATA.mean(), research.load_results().df.test_loss[0])
