"""
object_api_python.py

Pure-python implementation of models described by the classes in object_api.py.

"""

import copy
from functools import partial
import sys
import math
import random

import networkx as nx

import nengo.object_api as API


class SimulatorException(Exception):
    pass

class ResetIncomplete(SimulatorException):
    pass

class StepIncomplete(SimulatorException):
    pass

class SizeError(SimulatorException):
    """Implementation produced output of incorrect size"""


def sizeof(obj):
    try:
        return obj.size
    except AttributeError:
        pass

    try:
        return len(obj)
    except TypeError:
        pass

    if isinstance(obj, (int, float)):
        return 1

    raise TypeError(obj)


class MultiArray(object):
    def __init__(self, shape):
        self.shape = shape

    @property
    def size(self):
        return reduce(lambda x, y: x * y, self.shape, 1)

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return self.shape[0]


class Array1(MultiArray):
    def __init__(self, data):
        MultiArray.__init__(self, (len(data),))
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val


class Array2(MultiArray):
    def __init__(self, data, shape):
        rows, cols = shape
        MultiArray.__init__(self, shape)
        self.data = data
        self.arr = []
        self.rows, self.cols = shape
        assert len(data) == rows * cols
        for ii in xrange(rows):
            self.arr.append(data[ii * cols: (ii + 1) * cols])

    def __iter__(self): # numpy-like
        return iter(self.arr)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.arr[idx]
        else:
            raise NotImplementedError()

    def __setitem__(self, idx, val):
        if isinstance(idx, int):
            if len(val) != self.shape[1]:
                raise ValueError('wrong number of columns')
            self.arr[idx] = val
        else:
            raise NotImplementedError()


def dot(a, b):
    if isinstance(a, Array2) and isinstance(b, Array1):
        rval = []
        for rr in xrange(a.shape[0]):
            ksum = 0
            a_rr = a[rr]
            for cc in xrange(a.shape[1]):
                ksum += a_rr[cc] * b[cc]
            rval.append(ksum)
        return Array1(rval)
    else:
        raise NotImplementedError()

def outer(a, b):
    if isinstance(a, Array1) and isinstance(b, Array1):
        rval = []
        for rr in xrange(a.shape[0]):
            for cc in xrange(b.shape[0]):
                rval.append(a[rr] * b[cc])
        return Array2(rval, (a.shape[0], b.shape[0]))
    else:
        raise NotImplementedError()


def sub(a, b):
    if isinstance(a, Array1) and isinstance(b, Array1):
        if len(a) != len(b):
            raise ValueError()
        return Array1([ai - bi for ai, bi in zip(a, b)])
    elif isinstance(a, (float, int)) and isinstance(b, Array1):
        return Array1([a - bi for bi in b])
    else:
        raise NotImplementedError('sub', (type(a), type(b)))

def add(a, b):
    if isinstance(a, Array1) and isinstance(b, Array1):
        if len(a) != len(b):
            raise ValueError()
        return Array1([ai + bi for ai, bi in zip(a, b)])
    elif isinstance(a, Array2) and isinstance(b, Array2):
        if a.shape != b.shape:
            raise ValueError()
        return Array2([ai + bi for ai, bi in zip(a.data, b.data)], a.shape)
    elif isinstance(a, (float, int)) and isinstance(b, Array1):
        return Array1([a + bi for bi in b])
    else:
        raise NotImplementedError('add', (type(a), type(b)))


def mul_scalar(a, b):
    if isinstance(a, Array1):
        return Array1([ai * b for ai in a])
    elif isinstance(a, Array2):
        return Array2([ai * b for ai in a.data], a.shape)
    else:
        raise NotImplementedError()

def pow_scalar(a, b):
    if isinstance(a, Array1):
        return Array1([ai ** b for ai in a])
    elif isinstance(a, Array2):
        return Array2([ai ** b for ai in a.data], a.shape)
    else:
        raise NotImplementedError()



def scheduling_graph(network):
    all_members = network.all_members
    DG = nx.DiGraph()
    producer = {}
    for member in all_members:
        for key, val in member.inputs.items():
            if val is None:
                pass
            elif isinstance(val, API.DelayedVar):
                pass
            else:
                DG.add_edge(val, member)
        for key, val in member.outputs.items():
            if val is None:
                raise ValueError('Output cannot be None', key)
            elif isinstance(val, API.DelayedVar):
                raise ValueError('Output cannot be delayed', key)
            else:
                if producer.setdefault(val, member) is not member:
                    raise API.MultipleSourceError(val)
                DG.add_edge(member, val)
    return DG


class ImplBase(object):
    @staticmethod
    def build(obj, state, dt):
        pass
    @staticmethod
    def reset(obj, state):
        pass
    @staticmethod
    def step(obj, state):
        pass
    @staticmethod
    def probe(obj, state):
        pass
    @staticmethod
    def sample(obj, N):
        pass


class SimulatorState(object):
    def __init__(self):
        self.new_state = {}
        self.old_state = {}
        self.probes = {}

    def __contains__(self, item):
        return item in self.new_state

    def __getitem__(self, item):
        try:
            return self.old_state[item.var]
        except AttributeError:
            return self.new_state[item]

    def __setitem__(self, item, val):
        self.new_state[item] = val

    def step(self):
        self.old_state = self.new_state
        self.new_state = {}



class Simulator(API.SimulatorBase):

    impl_registry = {}
    build_registry = {}
    reset_registry = {}
    step_registry = {}
    sample_registry = {}

    @classmethod
    def register_impl(mycls, cls):
        api_cls = getattr(API, cls.__name__)
        mycls.build_registry[api_cls] = cls.build
        mycls.reset_registry[api_cls] = cls.reset
        mycls.step_registry[api_cls] = cls.step
        mycls.sample_registry[api_cls] = cls.sample
        return cls

    def __init__(self, network, dt, verbosity=0):
        API.SimulatorBase.__init__(self, network, dt)
        self.state = SimulatorState()
        self.verbosity = verbosity
        self.state[API.simulation_time] = self.simulation_time
        DG = scheduling_graph(network)
        #import matplotlib.pyplot as plt
        #nx.draw(DG)
        #plt.show()
        try:
            self.full_ordering = nx.topological_sort(DG)
        except nx.NetworkXUnfeasible:
            #nx.write_multiline_adjlist(DG, sys.stdout)
            raise API.SelfDependencyError()
        self.member_ordering = [n for n in self.full_ordering
                              if not isinstance(n, API.Var)]

        for member in self.member_ordering:
            build_fn = self.build_registry.get(type(member), None)
            if build_fn:
                if verbosity:
                    print 'Build:', member, build_fn, verbosity
                build_fn(member, self.state, self.dt)
            elif verbosity:
                print 'No build:', member
        self.reset()

    def reset(self):
        API.SimulatorBase.reset(self)
        self.state[API.simulation_time] = self.simulation_time
        for member in self.member_ordering:
            reset_fn = self.reset_registry.get(type(member), None)
            if reset_fn:
                if self.verbosity:
                    print 'Reset:', member, reset_fn
                reset_fn(member, self.state)
            elif self.verbosity:
                print 'No reset:', member
            for key, val in member.outputs.items():
                if val not in self.state:
                    raise ResetIncomplete("Reset %s did not produce outputs[%s]" % (
                        member, key))

    def run_steps(self, steps):
        step_fns = []
        for member in self.member_ordering:
            step_fn = self.step_registry.get(type(member), None)
            if step_fn:
                if self.verbosity:
                    print 'Step:', member, step_fn
                step_fns.append((step_fn, member))
            elif self.verbosity:
                print 'No step:', member

        for tt in xrange(steps):
            self.simulation_steps += 1
            self.state.step()
            self.state[API.simulation_time] = self.simulation_time
            for step_fn, member in step_fns:
                try:
                    step_fn(member, self.state)
                except TypeError, e:
                    if "takes exactly" in str(e):
                        e.args = e.args + (step_fn, 'of implementation for',  member)
                    raise
                for key, var in member.outputs.items():
                    if var not in self.state:
                        raise StepIncomplete("Step %s did not produce outputs[%s]" % (
                            member, key))

                    val = self.state[var]
                    if sizeof(val) != var.size:
                        raise SizeError(
                            "Step %s produced outputs[%s] of wrong size %i, which"
                            " should have been %i" % (
                            member, key, sizeof(val), var.size))

        rval = {}
        for probe in self.network.all_probes:
            # -- probe_fn is mandatory
            rval[probe.target] = self.state.probes[probe]
            self.state.probes[probe] = []

        return rval

    def run(self, sim_time):
        steps = int(sim_time / self.dt)
        return self.run_steps(steps)

API.SimulatorBase._backends['reference'] = Simulator
register_impl = Simulator.register_impl


#
# Pure Python Object API implementation
#

@register_impl
class TimeNode(ImplBase):
    @staticmethod
    def build(node, state, dt):
        state[node.output] = node.func(state[API.simulation_time])

    @staticmethod
    def reset(node, state):
        state[node.output] = node.func(state[API.simulation_time])

    @staticmethod
    def step(node, state):
        t = state[API.simulation_time]
        state[node.output] = node.func(t)


@register_impl
class Probe(ImplBase):
    @staticmethod
    def reset(probe, state):
        state.probes[probe] = []

    @staticmethod
    def step(probe, state):
        obj = copy.deepcopy(state[probe.target])
        state.probes[probe].append(obj)


def draw(dist, rng, N):
    if dist.dist_name == 'uniform':
        return [rng.uniform(dist.low, dist.high) for ii in xrange(N)]
    elif dist.dist_name == 'gaussian':
        return [rng.gauss(dist.mean, dist.std) for ii in xrange(N)]
    else:
        raise NotImplementedError()


@register_impl
class LIFNeurons(ImplBase):
    @staticmethod
    def build(neurons, state, dt):

        max_rates = draw(neurons.max_rate,
                random.Random(neurons.max_rate.seed), neurons.size)
        threshold = draw(neurons.intercept,
                random.Random(neurons.intercept.seed), neurons.size)

        def x_fn(max_rate):
            u = neurons.tau_ref - (1.0 / max_rate)
            return 1.0 / (1 - math.exp(u / neurons.tau_rc))
        xlist = map(x_fn, max_rates)
        alpha = [(1 - x) / intercept for x, intercept in zip(xlist, threshold)]
        j_bias = [1 - aa * intercept for aa, intercept in zip(alpha, threshold)]

        state[neurons.alpha] = Array1(alpha)
        state[neurons.j_bias] = Array1(j_bias)
        state[neurons.voltage] = Array1([0] * neurons.size)
        state[neurons.refractory_time] = Array1([0] * neurons.size)
        state[neurons.output] = Array1([0] * neurons.size)

    @staticmethod
    def reset(neurons, state):
        state[neurons.voltage] = Array1([0] * neurons.size)
        state[neurons.refractory_time] = Array1([0] * neurons.size)
        state[neurons.output] = Array1([0] * neurons.size)

    @staticmethod
    def step(neurons, state):
        alpha = state[neurons.inputs['alpha']]
        j_bias = state[neurons.inputs['j_bias']]
        voltage = state[neurons.inputs['voltage']]
        refractory_time = state[neurons.inputs['refractory_time']]
        try:
            input_current = state[neurons.inputs['input_current']]
        except KeyError:
            input_current = [0] * neurons.size

        J = [jbi + ici for jbi, ici in zip(j_bias, input_current)]
        assert len(J) == neurons.size
        tau_rc = neurons.tau_rc
        tau_ref = neurons.tau_ref
        dt = state[API.simulation_time] - state[API.simulation_time.delayed()]
        
        new_voltage = Array1([0] * neurons.size)
        new_refractory_time = Array1([0] * neurons.size)
        new_output = Array1([0] * neurons.size)

        def clip(a, low, high):
            if a < low:
                return low
            if a > high:
                return high
            return a

        for ii in xrange(neurons.size):

            # Euler's method
            dV = dt / tau_rc * (J[ii] - voltage[ii])

            # increase the voltage, ignore values below 0
            v = max(voltage[ii] + dV, 0)

            # handle refractory period
            post_ref = 1.0 - (refractory_time[ii] - dt) / dt

            # set any post_ref elements < 0 = 0, and > 1 = 1
            if post_ref < 0:
                v = 0
            elif post_ref < 1:
                v *= post_ref

            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked = 1 if v > 1 else 0

            # adjust refractory time (neurons that spike get
            # a new refractory time set, all others get it reduced by dt)

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            if spiked:
                new_refractory_time[ii] = spiketime + tau_ref
            else:
                new_refractory_time[ii] = refractory_time[ii] - dt

            new_voltage[ii] = v * (1 - spiked)
            new_output[ii] = spiked


        state[neurons.outputs['alpha']] = alpha
        state[neurons.outputs['j_bias']] = j_bias
        state[neurons.outputs['voltage']] = new_voltage
        state[neurons.outputs['refractory_time']] = new_refractory_time
        state[neurons.outputs['X']] = new_output


@register_impl
class Connection(ImplBase):
    @staticmethod
    def reset(self, state):
        src = state[self.inputs['X']]
        dst = src.copy()
        state[self.outputs['X']] = dst

    @staticmethod
    def step(self, state):
        src = state[self.inputs['X']]
        dst = src.copy()
        state[self.outputs['X']] = dst


@register_impl
class RandomConnection(ImplBase):
    @staticmethod
    def reset(self, state):
        n_in = self.src.size
        n_out = self.dst.size
        weights = draw(self.dist, random.Random(self.dist.seed),
                       n_out * n_in)

        state[self.outputs['X']] = Array1([0] * n_out)
        state[self.outputs['weights']] = Array2(weights, (n_out, n_in))

    @staticmethod
    def step(self, state):
        src = state[self.inputs['X']]
        weights = state[self.inputs['weights']]
        state[self.output] = dot(weights, src)
        state[self.outputs['weights']] = weights

@register_impl
class MSE_MinimizingConnection(ImplBase):
    @staticmethod
    def reset(self, state):
        n_in = self.src.size
        n_out = self.dst.size
        weights = Array2([0] * n_out * n_in, (n_out, n_in))

        state[self.outputs['X']] = Array1([0] * n_out)
        state[self.outputs['error_signal']] = Array1([0] * n_out)
        state[self.outputs['weights']] = weights

    @staticmethod
    def step(self, state):
        src = state[self.inputs['X']]
        target = state[self.inputs['target']]
        weights = state[self.inputs['weights']]

        prediction = dot(weights, src)

        grad = sub(target, prediction)
        weights = add(weights, mul_scalar(outer(grad, src), self.learning_rate))

        state[self.outputs['X']] = prediction
        state[self.outputs['error_signal']] = sum(pow_scalar(grad, 2).data)
        state[self.outputs['weights']] = weights


@register_impl
class Filter(ImplBase):
    @staticmethod
    def reset(self, state):
        state[self.output] = Array1([0] * self.output.size)

    @staticmethod
    def step(self, state):
        X_prev = state[self.inputs['X_prev']]
        var = state[self.inputs['var']]
        assert len(X_prev) == len(var) == self.output.size
        state[self.output] = Array1(
            [xi + self.tau * vi for xi, vi in zip(X_prev, var)])
