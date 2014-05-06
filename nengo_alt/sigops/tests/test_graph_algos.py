import pytest

from sigops import graph_algos

def test_reversedict():
    edges = graph_algos.graph({'a': set(['b', 'c'])})
    r_edges = graph_algos.reverse_edges(edges)
    assert r_edges == {'b': ('a',), 'c': ('a',)}


def test_toposort():
    edges = graph_algos.graph({'a': set(['b', 'c']), 'b': ('c',)})
    assert graph_algos.toposort(edges) == ['a', 'b', 'c']


def test_add_edges():
    edges = graph_algos.graph({'a': set(['b', 'c'])})
    graph_algos.add_edges(edges, [('a', 'd'), ('b', 'c')])
    assert edges == {'a': set(['b', 'c', 'd']), 'b': set(['c'])}

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
