import pyopencl as cl

class Plan(object):

    def __init__(self, dct):
        self.__dict__.update(dct)

        self._fn.set_args(*self._fn_args[3:])
        self._enqueue_args = (
            self.queue,
            self._fn,
            self._fn_args[1],
            self._fn_args[2],
            )

    def __call__(self):
        self._fn(*self._fn_args)
        self._fn_args[0].finish()


class Prog(object):
    def __init__(self, plans):
        self.plans = plans
        self.queues = [p._enqueue_args[0] for p in self.plans]
        self.kerns = [p._enqueue_args[1] for p in self.plans]
        self.gsize = [p._enqueue_args[2] for p in self.plans]
        self.lsize = [p._enqueue_args[3] for p in self.plans]

    def __call__(self):
        map(cl.enqueue_nd_range_kernel,
            self.queues, self.kerns, self.gsize, self.lsize)
        self.queues[-1].finish()
