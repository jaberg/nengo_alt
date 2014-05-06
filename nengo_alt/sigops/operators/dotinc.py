import numpy as np

from ..operator import Operator


def reshape_dot(A, X, Y, tag=None):
    """Checks if the dot product needs to be reshaped.

    Also does a bunch of error checking based on the shapes of A and X.
    """
    badshape = False
    ashape = (1,) if A.shape == () else A.shape
    xshape = (1,) if X.shape == () else X.shape
    if A.shape == ():
        incshape = X.shape
    elif X.shape == ():
        incshape = A.shape
    elif X.ndim == 1:
        badshape = ashape[-1] != xshape[0]
        incshape = ashape[:-1]
    else:
        badshape = ashape[-1] != xshape[-2]
        incshape = ashape[:-1] + xshape[:-2] + xshape[-1:]

    if (badshape or incshape != Y.shape) and incshape != ():
        raise ValueError('shape mismatch in %s: %s x %s -> %s' % (
            tag, A.shape, X.shape, Y.shape))

    # If the result is scalar, we'll reshape it so Y[...] += inc works
    return incshape == ()


class DotInc(Operator):
    """Increment signal Y by dot(A, X)"""

    def __init__(self, A, X, Y, tag=None):
        self.A = A
        self.X = X
        self.Y = Y
        self.tag = tag

        self.reads = [self.A, self.X]
        self.incs = [self.Y]
        self.sets = []
        self.updates = []

    def __str__(self):
        return 'DotInc(%s, %s -> %s "%s")' % (
            str(self.A), str(self.X), str(self.Y), self.tag)

    def make_step(self, signals):
        X = signals[self.X]
        A = signals[self.A]
        Y = signals[self.Y]
        reshape = reshape_dot(A, X, Y, self.tag)

        def step():
            inc = np.dot(A, X)
            if reshape:
                inc = np.asarray(inc).reshape(Y.shape)
            Y[...] += inc
        return step
