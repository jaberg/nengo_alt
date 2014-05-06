import numpy as np
import pytest

import nengo


def test_io(tmpdir):
    tmpfile = str(tmpdir.join("model.pkl"))
    m1 = nengo.Network()
    with m1:
        sin = nengo.Node(output=np.sin)
        cons = nengo.Node(output=-.5)
        factors = nengo.Ensemble(nengo.LIF(20), dimensions=2, radius=1.5)
        factors.encoders = np.tile(
            [[1, 1], [-1, 1], [1, -1], [-1, -1]],
            (factors.n_neurons // 4, 1))
        product = nengo.Ensemble(nengo.LIFRate(10), dimensions=1)
        nengo.Connection(sin, factors[0])
        nengo.Connection(cons, factors[1])
        factors_p = nengo.Probe(
            factors, 'decoded_output', sample_every=.01, synapse=.01)
        assert factors_p  # To suppress F841
        product_p = nengo.Probe(
            product, 'decoded_output', sample_every=.01, synapse=.01)
        assert product_p  # To suppress F841
    m1.save(tmpfile)
    m2 = nengo.Network.load(tmpfile)
    assert m1 == m2


def test_basic_context(Simulator):
    # We must give the two Networks different labels because object comparison
    # is done using identifiers that stem from the top-level Network label.
    model1 = nengo.Network(label="test1")
    model2 = nengo.Network(label="test2")

    with model1:
        e = nengo.Ensemble(nengo.LIF(1), 1)
        n = nengo.Node([0])
        assert e in model1.ensembles
        assert n in model1.nodes

        con = nengo.Network()
        with con:
            e2 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e2 in con.ensembles
        assert e2 not in model1.ensembles

        e3 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e3 in model1.ensembles

    with model2:
        e4 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e4 not in model1.ensembles
        assert e4 in model2.ensembles


def test_nested_context(Simulator):
    model = nengo.Network()
    with model:
        con1 = nengo.Network()
        con2 = nengo.Network()
        con3 = nengo.Network()

        with con1:
            e1 = nengo.Ensemble(nengo.LIF(1), 1)
            assert e1 in con1.ensembles

            with con2:
                e2 = nengo.Ensemble(nengo.LIF(1), 1)
                assert e2 in con2.ensembles
                assert e2 not in con1.ensembles

                with con3:
                    e3 = nengo.Ensemble(nengo.LIF(1), 1)
                    assert e3 in con3.ensembles
                    assert e3 not in con2.ensembles \
                        and e3 not in con1.ensembles

                e4 = nengo.Ensemble(nengo.LIF(1), 1)
                assert e4 in con2.ensembles
                assert e4 not in con3.ensembles

            e5 = nengo.Ensemble(nengo.LIF(1), 1)
            assert e5 in con1.ensembles

        e6 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e6 not in con1.ensembles


def test_context_errors(Simulator):
    def add_something():
        nengo.Ensemble(nengo.LIF(1), 1)

    # Error if adding before Network creation
    with pytest.raises(RuntimeError):
        add_something()

    model = nengo.Network()
    # Error if adding before a `with network` block
    with pytest.raises(RuntimeError):
        add_something()

    # Error if adding after a `with network` block
    with model:
        add_something()
    with pytest.raises(RuntimeError):
        add_something()

    # Okay if add_to_container=False
    nengo.Ensemble(nengo.LIF(1), 1, add_to_container=False)
    nengo.Node(output=[0], add_to_container=False)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
