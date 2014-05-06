import time
import theano

import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
print cl.get_platforms()
print ctx.devices

from api2 import (
    OCL_LIFNeuron,
    random_low_rank_connection,
    Simulator,
    )

nengo_1s = {}
nengo_1s[(10, 10, 1)] = 0.802000045776
nengo_1s[(10, 10, 2)] = 0.666999816895
nengo_1s[(10, 10, 50)] = 1.08800005913
nengo_1s[(10, 100, 1)] = 1.15799999237
nengo_1s[(10, 100, 2)] = 1.37699985504
nengo_1s[(10, 100, 50)] = 1.72799992561

nengo_1s[(100, 10, 1)] = 1.28299999237
nengo_1s[(100, 10, 2)] = 1.36100006104
nengo_1s[(100, 10, 50)] = 3.16299986839
nengo_1s[(100, 100, 1)] = 6.14800000191
nengo_1s[(100, 100, 2)] = 6.14300012589
nengo_1s[(100, 100, 50)] = 9.89700007439

nengo_1s[(1000, 10, 1)] = 7.7009999752
nengo_1s[(1000, 10, 2)] = 8.11899995804
nengo_1s[(1000, 10, 50)] = 25.9370000362
nengo_1s[(1000, 100, 1)] = 50.736000061
nengo_1s[(1000, 100, 2)] = 52.0779998302
nengo_1s[(1000, 100, 50)] = 76.2179999352

# -- neuron dynamics are simulated at .0005
#    so we credit nengo with running the whole thing
#    at 200 steps / simulated second
nengo_1s_steps = 2000 # dt = 0.0005 seconds


#from .. import nef_theano as nef
for n_ensembles in [10, 100, 1000, 10000]:
    for size in [10, 100, 1000]:
        for rank in [1, 2, 50]:
            if n_ensembles * size * rank > 10 * 1000 * 1000:
                continue
            key = (n_ensembles, size, rank)
            simtime = 0.5

            p = OCL_LIFNeuron(queue, size=size * n_ensembles)
            pops = [p[ii * size:(ii + 1) * size]
                for ii in range(n_ensembles)]

            connections = [random_low_rank_connection(queue, p1, p2, rank)
                for p1, p2 in zip(pops[:-1], pops[1:])]

            sim = Simulator([p], connections)
            t0 = time.time()
            simtime = 0.1
            n_steps = int(2000 * simtime)
            sim.step(queue, n_steps, 1./2000)
            t1 = time.time()
            elapsed = t1 - t0
            our_walltime = (t1 - t0) / simtime
            print n_ensembles, size, rank, 'walltime', our_walltime,
            print 'steps/sec', n_steps / elapsed,
            if key in nengo_1s:
                nengo_walltime = nengo_1s[key]
                print 'rel-to nengo:', nengo_walltime / our_walltime,
            print ''


