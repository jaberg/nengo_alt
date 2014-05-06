import logging

import numpy as np
import pytest

from nengo_alt import sigops
from nengo_alt.sigops.operators import (
    ProdUpdate, Copy, Reset, DotInc)
from nengo_alt.sigops import Signal

logger = logging.getLogger(__name__)


def pytest_funcarg__Engine(request):
    return sigops.Engine


def test_signal_init_values(Engine):
    """Tests that initial values are not overwritten."""
    zero = Signal([0])
    one = Signal([1])
    five = Signal([5.0])
    zeroarray = Signal([[0], [0], [0]])
    array = Signal([1, 2, 3])

    operators = [ProdUpdate(zero, zero, one, five),
                 ProdUpdate(zeroarray, one, one, array)]

    engine = Engine(operators)
    assert engine.signals[zero][0] == 0
    assert engine.signals[one][0] == 1
    assert engine.signals[five][0] == 5.0
    assert np.all(np.array([1, 2, 3]) == engine.signals[array])
    engine.step()
    assert engine.signals[zero][0] == 0
    assert engine.signals[one][0] == 1
    assert engine.signals[five][0] == 5.0
    assert np.all(np.array([1, 2, 3]) == engine.signals[array])


def test_steps(Engine):
    engine = Engine([])
    assert engine.n_steps == 0
    engine.step()
    assert engine.n_steps == 1
    engine.step()
    assert engine.n_steps == 2
    engine.step(N=5)
    assert engine.n_steps == 7



def test_signal_indexing_1(Engine):
    one = Signal(np.zeros(1), name="a")
    two = Signal(np.zeros(2), name="b")
    three = Signal(np.zeros(3), name="c")
    tmp = Signal(np.zeros(3), name="tmp")

    operators = [
        ProdUpdate(
            Signal(1, name="A1"), three[:1],
            Signal(0, name="Z0"), one),
        ProdUpdate(
            Signal(2.0, name="A2"), three[1:],
            Signal(0, name="Z1"), two),
        Reset(tmp),
        DotInc(
            Signal([[0, 0, 1], [0, 1, 0], [1, 0, 0]], name="A3"),
            three, tmp),
        Copy(src=tmp, dst=three, as_update=True),
    ]

    engine = Engine(operators)
    engine.signals[three] = np.asarray([1, 2, 3])
    engine.step()
    assert np.all(engine.signals[one] == 1)
    assert np.all(engine.signals[two] == [4, 6])
    assert np.all(engine.signals[three] == [3, 2, 1])
    engine.step()
    assert np.all(engine.signals[one] == 3)
    assert np.all(engine.signals[two] == [4, 2])
    assert np.all(engine.signals[three] == [1, 2, 3])



if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, "-v"])
