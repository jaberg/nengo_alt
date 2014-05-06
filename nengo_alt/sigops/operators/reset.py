from ..operator import Operator


class Reset(Operator):
    """Assign a constant value to a Signal."""

    def __init__(self, dst, value=0):
        self.dst = dst
        self.value = float(value)

        self.reads = []
        self.incs = []
        self.updates = []
        self.sets = [dst]

    def __str__(self):
        return 'Reset(%s)' % str(self.dst)

    def make_step(self, signals):
        target = signals[self.dst]
        value = self.value

        def step():
            target[...] = value
        return step

