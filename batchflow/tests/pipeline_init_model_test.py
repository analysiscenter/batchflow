""" Tests for pipeline `init_model`  action """
# pylint: disable=import-error, no-name-in-module, redefined-outer-name
import pytest

from batchflow import Dataset, C
from batchflow.models import BaseModel

@pytest.fixture
def dataset():
    """ Toy dataset """
    return Dataset(100)

@pytest.mark.parametrize('use_pipeline_algebra', [True, False])
@pytest.mark.parametrize('config', [dict(base_class=BaseModel, name='model')])
def test_single_model(use_pipeline_algebra, config, dataset):
    """ Verifies that the model initialized in `init_model` method is stored in the pipeline
    with expected name and belongs to the expected class in case name and base_class are passsed
    as C() named expressions to the method. Verifies that strategies of linking the config to
    the pipeline via pipeline algebra or manually are equal.
    """
    if use_pipeline_algebra:
        pipeline = dataset.pipeline().init_model('static', C('base_class'), C('name'), {}) << config
    else:
        pipeline = dataset.pipeline(config).init_model('static', C('base_class'), C('name'), {})
    pipeline.next_batch(1)

    assert pipeline.m(config['name']) # the model with fiven name exist
    assert isinstance(pipeline.m(config['name']), config['base_class']) # the model belongs to expected class

@pytest.mark.parametrize('use_pipeline_algebra', [True, False])
@pytest.mark.parametrize('config1', [dict(name='model1')])
@pytest.mark.parametrize('config2', [dict(name='model2')])
def test_multiple_models(use_pipeline_algebra, config1, config2, dataset):
    """ Verifies that several models initialized in `init_model` via sequentially linking different
    configs with the same keys but differnt values to the pipeline are properly stored in the pipeline
    and not overwriting each other from linkage to linkage.
    Checks that 2 models are stored with expected names.
    """
    if not use_pipeline_algebra:
        pipeline = dataset.pipeline().init_model('static', BaseModel, C('name'), {}) << config1
    else:
        pipeline = dataset.pipeline(config1).init_model('static', BaseModel, C('name'), {})
    pipeline.next_batch(1)

    pipeline = pipeline << config2
    pipeline.next_batch(1)

    assert pipeline.m(config1['name']) and pipeline.m(config2['name'])
