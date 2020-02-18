import pytest 

from batchflow import DatasetIndex, Pipeline, Batch, C
from batchflow.models import BaseModel
from batchflow.model_dir import NonInitializedModel

@pytest.fixture
def dataset():
    index = DatasetIndex(100)
    return Dataset(index, Batch)

@pytest.mark.parametrize('use_algebra', [True, False])
@pytest.mark.parametrize('config', [dict(mode='static', base_class=BaseModel, name='model', config={}),
                                    dict(mode='dynamic', base_class=BaseModel, name='model', config={})])
def test_single_model(use_algebra, config, dataset):
    if not use_algebra:
        pipeline = dataset.pipeline(config).init_model(C('mode'), C('base_class'), C('name'), C('config'))
    else:
        pipeline = dataset.pipeline().init_model(C('mode'), C('class'), C('base_name'), C('config')) << config
    pipeline.next_batch(1)

    if config['mode'] == 'static': 
        assert type(pipeline.m(config['name']) == config['class']
    else:
        assert type(pipeline.m(config['name']) == NonInitializedModel

# @pytest.mark.parametrize('use_algebra1', [True, False])
# @pytest.mark.parametrize('use_algebra2', [True, False])
# @pytest.mark.parametrize('mode', ['static', 'dynamic'])
# @pytest.mark.parametrize('config1', 'config2' [(dict(class=BaseModel, name='model1', config={}),
#                                                 dict(class=BaseModel, name='model2', config={})])
# def test_multiple_models(dataset, use_algebra1, use_algebra2, mode,  config1, config2):
#     config1.update(mode=mode)
#     if not use_algebra1:
#         pipeline = dataset.pipeline(config1).init_model(C('mode'), C('class'), C('name'), C('config'))
#     else:
#         pipeline = dataset.p.init_model(C('mode'), C('class'), C('name'), C('config')) << config1
#     pipeline.next_batch(1)

#     config2.update(mode=mode)
#     if not use_algebra2:
#         pipeline = pipeline.update_config(config2)
#     else:
#         pipeline = pipeline << config2
#     pipeline.next_batch(1)

#     assert type(pipeline.m(config1['name']) == type(pipeline.m(config2['name'])
