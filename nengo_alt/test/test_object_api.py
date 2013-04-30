import math
import unittest

import nose.tools

from nengo import object_api as API
from nengo.object_api import (
    Connection,
    Filter,
    LIFNeurons,
    MSE_MinimizingConnection,
    Network,
    Probe,
    simulation_time,
    SelfDependencyError,
    TimeNode,
    )

class SmokeTests(unittest.TestCase):

    def Simulator(self, *args, **kwargs):
        return API.Simulator(backend='reference', *args, **kwargs)

    def test_smoke_1(self):
        net = Network()
        tn = net.add(TimeNode(math.sin, name='sin'))
        net.add(Probe(tn.output))
        net.add(Probe(simulation_time))
        sim = self.Simulator(net, dt=0.001, verbosity=0)
        results = sim.run_steps(101)

        assert len(results[simulation_time]) == 101
        for i, t in enumerate(results[simulation_time]):
            assert results[tn.output][i] == math.sin(t)

    def test_smoke_2(self):
        net = Network()
        net.add(Probe(simulation_time))

        ens = net.add(LIFNeurons(13))
        net.add(Probe(ens.output))

        sim = self.Simulator(net, dt=0.001, verbosity=1)
        results = sim.run_steps(101)

        assert len(results[simulation_time]) == 101
        total_n_spikes = 0
        for i, t in enumerate(results[simulation_time]):
            output_i = results[ens.output][i]
            assert len(output_i) == 13
            assert all(oi in (0, 1) for oi in output_i)
            total_n_spikes += sum(output_i)
        assert total_n_spikes > 0

    def test_schedule_self_dependency(self):
        net = Network()
        net.add(Probe(simulation_time))

        v1 = API.Var(size=1)
        v2 = API.Var(size=1)
        c1 = net.add(Connection(v1, v2))
        c2 = net.add(Connection(v2, v1))

        nose.tools.assert_raises(SelfDependencyError,
            self.Simulator, net, dt=0.001, verbosity=0)

    def test_schedule(self):
        net = Network()

        ens1 = net.add(LIFNeurons(3))
        v = API.Var(size=3)
        c1 = net.add(Connection(ens1.output, v))
        filt = net.add(Filter(v))
        probe = net.add(Probe(filt.output))
        assert ens1.output is not None

        sim = self.Simulator(net, dt=0.001, verbosity=0)
        ordering = sim.member_ordering
        desired = [ens1, c1, filt, probe]
        assert desired == ordering, (desired, ordering)

    def test_schedule_cycle_with_filter(self):
        net = Network()
        ens1 = net.add(LIFNeurons(13))
        filt = net.add(Filter(ens1.output.delayed()))
        conn = net.add(Connection(filt.output, ens1.input_current))
        net.add(Probe(filt.output))
        sim = self.Simulator(net, dt=0.001, verbosity=1)
        results = sim.run(.1)

    def test_repeatable(self):
        net = Network()
        ens1 = net.add(LIFNeurons(13))
        filter = net.add(Filter(ens1.output))
        conn = net.add(Connection(filter.output, ens1.input_current))
        net.add(Probe(filter.output))
        sim = self.Simulator(net, dt=0.001, verbosity=0)
        results = sim.run(.1)
        raise nose.SkipTest()
        #sim2 = self.Simulator(net, dt=0.001, verbosity=0)
        #results2 = sim.run(.1)
        # -- this is not the right way to check because array equality
        #    is ambiguous
        #assert results == results2

    def test_fan_in(self):
        net = Network()
        ens1 = net.add(LIFNeurons(1))
        ens2 = net.add(LIFNeurons(2))
        ens3 = net.add(LIFNeurons(3))
        conn = net.add(Connection(ens1.output, ens3.input_current))
        conn = net.add(Connection(ens2.output, ens3.input_current))

        nose.tools.assert_raises(API.MultipleSourceError,
            self.Simulator, net, dt=0.001, verbosity=0)

    def test_learning_rate(self, N=100, cls=API.LIFRateNeurons, lr=1e-4):
        net = Network()
        tn = net.add(TimeNode(math.sin, name='sin'))

        ens1 = net.add(cls(N))

        conn = net.add(
            MSE_MinimizingConnection(
                ens1.output,
                API.Var(size=1),
                target=tn.output,
                learning_rate=lr))
        rec = net.add(
            API.RandomConnection(
                tn.output,
                ens1.input_current,
                API.Uniform(-20.010, 20.010, seed=123)
                ))

        filt = net.add(API.Filter(conn.output, .01))

        net.add(Probe(conn.output))
        net.add(Probe(filt.output))
        net.add(Probe(tn.output))
        net.add(Probe(ens1.input_current))
        net.add(Probe(ens1.output))

        sim = self.Simulator(net, dt=0.001, verbosity=0)

        import matplotlib.pyplot as plt
        import numpy as np

        iters = 30
        for i in range(iters):
            last = i == iters - 1
            if last:
                conn.learning_rate = 0
            results = sim.run(2 * math.pi)
            # -- tests that the learned connection does better than predict
            #    mean 0
            pred_out = results[conn.output]
            filt_out = results[filt.output]
            want_out = results[tn.output]
            neur_out = np.asarray(results[ens1.output])
            if 0 and ((i % 5 == 0) or last):
                plt.plot(pred_out)
                plt.plot(want_out)
                plt.plot(filt_out)
                for i in range(20):
                    plt.plot(neur_out[:, i])
                plt.show()

            mse = sum([(po - wo) ** 2 for po, wo in zip(pred_out, want_out)])
            print mse

        assert mse < 10

        #ens1 = net.add(API.LIFNeurons(N))

    def test_learning_spiking(self, N=100):
        return self.test_learning_rate(N=N, cls=API.LIFNeurons, lr=1e-4)
