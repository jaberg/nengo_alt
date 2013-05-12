import time
import numpy as np

import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

from api3 import LIFMultiEnsemble

#from .. import nef_theano as nef
for n_ensembles in [10, 100, 1000, 10000]:
    for size in [10, 100, 1000]:
        for rank in [1, 2, 50]:
            if n_ensembles * size * rank >= 10 * 1000 * 1000:
                continue
            key = (n_ensembles, size, rank)
            simtime = 0.5

            mdl = LIFMultiEnsemble(
                    n_populations=n_ensembles,
                    n_neurons_per=size,
                    n_signals=n_ensembles,
                    signal_size=rank,
                    queue=queue)

            # TODO: set up the weights to do a real fn
            mdl._randomize_decoders(rng=np.random)
            mdl._randomize_encoders(rng=np.random)
            prog = mdl.prog(dt=0.001)

            prog() # once for allocation & warmup
            t0 = time.time()
            for i in range(1000):
                map(cl.enqueue_nd_range_kernel,
                    prog.queues, prog.kerns, prog.gsize, prog.lsize)
            queue.finish()
            t1 = time.time()
            simsec = (t1 - t0)
            print '%8i %8i %8i: %i neurons took %s seconds' %(
                    n_ensembles, size, rank, n_ensembles * size, simsec)

