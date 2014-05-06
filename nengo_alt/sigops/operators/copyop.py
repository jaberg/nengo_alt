from ..operator import Operator


class Copy(Operator):
    """Assign the value of one signal to another."""

    def __init__(self, dst, src, as_update=False, tag=None):
        self.dst = dst
        self.src = src
        self.tag = tag
        self.as_update = True

        self.reads = [src]
        self.sets = [] if as_update else [dst]
        self.updates = [dst] if as_update else []
        self.incs = []

    def __str__(self):
        return 'Copy(%s -> %s, as_update=%s)' % (
            str(self.src), str(self.dst), self.as_update)

    def make_step(self, signals):
        dst = signals[self.dst]
        src = signals[self.src]

        def step():
            dst[...] = src
        return step

