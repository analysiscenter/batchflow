import os
import dill
import pytest

from batchflow import Dataset, Pipeline, V
from batchflow.research import Option, Research

PPL_0 = (Dataset(10).p
         .init_variable('a', 0)
         .init_variable('b', 0)
         .update(V('a'), V('a')+1)
         .update(V('b'), V('b')+2)
         .run_later(1)
)

PPL_1 = (Dataset(20).p
         .init_variable('c', 0)
         .update(V('c'), V('c')+3)
         .run_later(1, n_epochs=1)
)

def test_results(tmp_path):
    path = os.path.join(tmp_path, 'research')
    research = (Research()
                .add_pipeline(PPL_0, variables=['a', 'b'], name='ppl_0')
                .add_pipeline(PPL_1, variables='c', name='ppl_1', run=True, execute='last'))
    research.run(n_iters=10, name=path)

    df = research.load_results().df

    assert len(df) == 11
    assert df[(df.iteration == 9) & (df.name == 'ppl_0')].a.values[0] == 10
    assert df[(df.iteration == 9) & (df.name == 'ppl_0')].b.values[0] == 20
    assert df[(df.iteration == 9) & (df.name == 'ppl_1')].c.values[0] == 60

    df = research.load_results(iterations=9).df
    assert df[df.name == 'ppl_0'].a.values[0] == 10
    assert df[df.name == 'ppl_0'].b.values[0] == 20
    assert df[df.name == 'ppl_1'].c.values[0] == 60

    df = research.load_results(iterations=9, names='ppl_1').df
    assert df.c.values[0] == 60

#     df = research.load_results().df
#     assert len(df) = 

#     df = research.load_results(iterations=[5, 8]).df
#     assert list(df.iteration) == [5, 8]

# def test_research(tmp_path):
#     path = os.path.join(tmp_path, 'research')
#     print('path', path)
#     research = (Research()
#                .add_pipeline(PPL_0, variables='counter', name='ppl_0')
#                .add_pipeline(PPL_1, variables='counter', name='ppl_1', run=True, execute='last')
#     )
#     research.run(name=path)

#     assert len(research.load_results().df) == 11
#     assert research.load_results(iterations=10, names='ppl_0').df.counter.values[0] == 10
#     assert research.load_results(iterations=10, names='ppl_1').df.counter.values[0] == 20