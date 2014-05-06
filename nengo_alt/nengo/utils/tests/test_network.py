import logging

import pytest

import nengo

logger = logging.getLogger(__name__)


def test_withself():
    model = nengo.Network(label='test_withself')
    with model:
        n1 = nengo.Node(output=0.5)
        assert n1 in model.nodes
        e1 = nengo.Ensemble(nengo.LIF(10), 1)
        assert e1 in model.ensembles
        c1 = nengo.Connection(n1, e1)
        assert c1 in model.connections
        ea1 = nengo.networks.EnsembleArray(nengo.LIF(10), 2)
        assert ea1 in model.networks
        assert len(ea1.ensembles) == 2
        n2 = ea1.add_output("out", None)
        assert n2 in ea1.nodes
        with ea1:
            e2 = nengo.Ensemble(nengo.LIF(10), 1)
            assert e2 in ea1.ensembles
    assert len(nengo.Network.context) == 0


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
