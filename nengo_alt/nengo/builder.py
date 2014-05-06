"""Reference implementation for building a nengo.Network."""

import collections
import logging

import numpy as np

import nengo.decoders
import nengo.neurons
import nengo.objects
import nengo.utils.distributions as dists
import nengo.utils.numpy as npext
from nengo.utils.compat import is_callable, is_integer

logger = logging.getLogger(__name__)


class ShapeMismatch(ValueError):
    pass

from sigops import Signal, Operator
from sigops.operators import Reset, Copy, DotInc, ProdUpdate


class SimPyFunc(Operator):
    """Set signal `output` by some non-linear function of x, possibly t"""

    def __init__(self, output, fn, t_in, x):
        self.output = output
        self.fn = fn
        self.t_in = t_in
        self.x = x

        self.reads = [] if x is None else [x]
        self.updates = [] if output is None else [output]
        self.sets = []
        self.incs = []

    def __str__(self):
        return "SimPyFunc(%s -> %s '%s')" % (self.x, self.output, self.fn)

    def make_step_dt(self, signals, dt):
        if self.output is not None:
            output = signals[self.output]
        fn = self.fn
        args = [signals['__time__']] if self.t_in else []
        args += [signals[self.x]] if self.x is not None else []

        def step():
            y = fn(*args)
            if self.output is not None:
                if y is None:
                    raise ValueError(
                        "Function '%s' returned invalid value" % fn.__name__)
                output[...] = y

        return step


class SimNeurons(Operator):
    """Set output to neuron model output for the given input current."""

    def __init__(self, neurons, J, output, states=[]):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.states = states

        self.reads = [J]
        self.updates = [output] + states
        self.sets = []
        self.incs = []

    def make_step_dt(self, signals, dt):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        def step():
            self.neurons.step_math(dt, J, output, *states)
        return step


class Model(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt=0.001, label=None, seed=None):
        # Resources used by the build process.
        self.operators = []
        self.params = {}
        self.probes = []
        self.sig_in = {}
        self.sig_out = {}

        self.dt = dt
        self.label = label
        self.seed = np.random.randint(npext.maxint) if seed is None else seed

        self.rng = np.random.RandomState(self.seed)

    def __str__(self):
        return "Model: %s" % self.label

    def has_built(self, obj):
        """Returns true iff obj has been processed by build."""
        return obj in self.params

    def next_seed(self):
        """Yields a seed to use for RNG during build computations."""
        return self.rng.randint(npext.maxint)


BuiltConnection = collections.namedtuple(
    'BuiltConnection', ['decoders', 'eval_points', 'transform', 'solver_info'])
BuiltNeurons = collections.namedtuple('BuiltNeurons', ['gain', 'bias'])
BuiltEnsemble = collections.namedtuple(
    'BuiltEnsemble',
    ['eval_points', 'encoders', 'intercepts', 'max_rates', 'scaled_encoders'])


class Builder(object):
    builders = {}

    @classmethod
    def register_builder(cls, build_fn, nengo_class):
        cls.builders[nengo_class] = build_fn

    @classmethod
    def build(cls, obj, *args, **kwargs):
        model = kwargs.setdefault('model', Model())

        if model.has_built(obj):
            # If we've already built the obj, we'll ignore it.
            # This is most likely the result of Neurons being used in
            # two different Ensembles, which is unlikely to be desired.

            # TODO: Prevent this at pre-build validation time.
            logger.warning("Object '%s' has already been built in model "
                           "'%s'." % (str(obj), model.label))
            return

        for obj_cls in obj.__class__.__mro__:
            if obj_cls in cls.builders:
                break
        else:
            raise TypeError("Cannot build object of type '%s'." %
                            cls.__name__)
        cls.builders[obj_cls](obj, *args, **kwargs)
        if obj not in model.params:
            raise RuntimeError(
                "Build function '%s' did not add '%s' to model.params"
                % (cls.builders[obj_cls].__name__, str(obj)))
        return model


def build_network(network, model):
    """Takes a Network object and returns a Model.

    This determines the signals and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

    1) Ensembles, Nodes, Neurons, Probes
    2) Subnetworks (recursively)
    3) Connections
    """
    logger.info("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        Builder.build(obj, model=model)

    logger.info("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        Builder.build(subnetwork, model=model)

    logger.info("Network step 3: Building connections")
    for conn in network.connections:
        Builder.build(conn, model=model)
    model.params[network] = None

Builder.register_builder(build_network, nengo.objects.Network)


def pick_eval_points(ens, n_points, rng):
    if n_points is None:
        # use a heuristic to pick the number of points
        dims, neurons = ens.dimensions, ens.neurons.n_neurons
        n_points = max(np.clip(500 * dims, 750, 2500), 2 * neurons)
    return dists.UniformHypersphere(ens.dimensions).sample(
        n_points, rng=rng) * ens.radius


def build_ensemble(ens, model):  # noqa: C901
    # Create random number generator
    seed = model.next_seed() if ens.seed is None else ens.seed
    rng = np.random.RandomState(seed)

    # Generate eval points
    if ens.eval_points is None or is_integer(ens.eval_points):
        eval_points = pick_eval_points(
            ens=ens, n_points=ens.eval_points, rng=rng)
    else:
        eval_points = npext.array(
            ens.eval_points, dtype=np.float64, min_dims=2)

    # Set up signal
    model.sig_in[ens] = Signal(np.zeros(ens.dimensions),
                               name="%s.signal" % ens.label)
    model.operators.append(Reset(model.sig_in[ens]))

    # Set up encoders
    if ens.encoders is None:
        if isinstance(ens.neurons, nengo.Direct):
            encoders = np.identity(ens.dimensions)
        else:
            sphere = dists.UniformHypersphere(ens.dimensions, surface=True)
            encoders = sphere.sample(ens.neurons.n_neurons, rng=rng)
    else:
        encoders = np.array(ens.encoders, dtype=np.float64)
        enc_shape = (ens.neurons.n_neurons, ens.dimensions)
        if encoders.shape != enc_shape:
            raise ShapeMismatch(
                "Encoder shape is %s. Should be (n_neurons, dimensions); "
                "in this case %s." % (encoders.shape, enc_shape))
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Determine max_rates and intercepts
    if isinstance(ens.max_rates, dists.Distribution):
        max_rates = ens.max_rates.sample(
            ens.neurons.n_neurons, rng=rng)
    else:
        max_rates = np.array(ens.max_rates)
    if isinstance(ens.intercepts, dists.Distribution):
        intercepts = ens.intercepts.sample(
            ens.neurons.n_neurons, rng=rng)
    else:
        intercepts = np.array(ens.intercepts)

    # Build the neurons
    if isinstance(ens.neurons, nengo.Direct):
        Builder.build(ens.neurons, ens.dimensions, model=model)
    else:
        Builder.build(ens.neurons, max_rates, intercepts, model=model)
    bn = model.params[ens.neurons]

    # Scale the encoders
    if isinstance(ens.neurons, nengo.Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (bn.gain / ens.radius)[:, np.newaxis]

    # Create output signal, using built Neurons
    model.operators.append(DotInc(
        Signal(scaled_encoders, name="%s.scaled_encoders" % ens.label),
        model.sig_in[ens],
        model.sig_in[ens.neurons],
        tag="%s encoding" % ens.label))

    # Output is neural output
    model.sig_out[ens] = model.sig_out[ens.neurons]

    for probe in ens.probes["decoded_output"]:
        Builder.build(probe, dimensions=ens.dimensions, model=model)
    for probe in ens.probes["spikes"] + ens.probes["voltages"]:
        Builder.build(probe, dimensions=ens.neurons.n_neurons, model=model)

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders)

Builder.register_builder(build_ensemble, nengo.objects.Ensemble)


def build_node(node, model):
    # Get input
    if node.output is None or is_callable(node.output):
        if node.size_in > 0:
            model.sig_in[node] = Signal(
                np.zeros(node.size_in), name="%s.signal" % node.label)
            # Reset input signal to 0 each timestep
            model.operators.append(Reset(model.sig_in[node]))

    # Provide output
    if node.output is None:
        model.sig_out[node] = model.sig_in[node]
    elif not is_callable(node.output):
        model.sig_out[node] = Signal(node.output, name=node.label)
    else:
        sig_in, sig_out = build_pyfunc(fn=node.output,
                                       t_in=True,
                                       n_in=node.size_in,
                                       n_out=node.size_out,
                                       label="%s.pyfn" % node.label,
                                       model=model)
        if sig_in is not None:
            model.operators.append(DotInc(
                model.sig_in[node],
                Signal(1.0, name="1"),
                sig_in,
                tag="%s input" % node.label))
        if sig_out is not None:
            model.sig_out[node] = sig_out

    for probe in node.probes["output"]:
        Builder.build(probe, dimensions=model.sig_out[node].shape, model=model)

    model.params[node] = None

Builder.register_builder(build_node, nengo.objects.Node)


def build_probe(probe, dimensions, model):
    model.sig_in[probe] = Signal(np.zeros(dimensions), name=probe.label)

    # Reset input signal to 0 each timestep
    model.operators.append(Reset(model.sig_in[probe]))
    model.probes.append(probe)

    # We put a list here so that the simulator can fill it
    # as it simulates the model
    model.params[probe] = []

Builder.register_builder(build_probe, nengo.objects.Probe)


def decay_coef(pstc, dt):
    pstc = max(pstc, dt)
    return np.exp(-dt / pstc)


def filtered_signal(signal, pstc, model):
    name = "%s.filtered(%f)" % (signal.name, pstc)
    filtered = Signal(np.zeros(signal.size), name=name)
    decay = decay_coef(pstc=pstc, dt=model.dt)
    model.operators.append(ProdUpdate(
        Signal(1.0 - decay, name="1 - decay"),
        signal,
        Signal(decay, name="decay"),
        filtered,
        tag="%s filtering" % name))
    return filtered


def build_connection(conn, model):  # noqa: C901
    rng = np.random.RandomState(model.next_seed())

    model.sig_in[conn] = model.sig_out[conn.pre]
    model.sig_out[conn] = model.sig_in[conn.post]

    decoders = None
    eval_points = None
    solver_info = None
    transform = np.array(conn.transform_full, dtype=np.float64)

    # Figure out the signal going across this connection
    if (isinstance(conn.pre, nengo.Ensemble)
            and isinstance(conn.pre.neurons, nengo.Direct)):
        # Decoded connection in directmode
        if conn.function is None:
            signal = model.sig_in[conn]
        else:
            sig_in, signal = build_pyfunc(
                fn=conn.function,
                t_in=False,
                n_in=model.sig_in[conn].size,
                n_out=conn.dimensions,
                label=conn.label,
                model=model)
            model.operators.append(DotInc(
                model.sig_in[conn],
                Signal(1.0, name="1"),
                sig_in,
                tag="%s input" % conn.label))
    elif isinstance(conn.pre, nengo.Ensemble):
        # Normal decoded connection
        encoders = model.params[conn.pre].encoders
        gain = model.params[conn.pre.neurons].gain
        bias = model.params[conn.pre.neurons].bias

        eval_points = conn.eval_points
        if eval_points is None:
            eval_points = npext.array(
                model.params[conn.pre].eval_points, min_dims=2)
        elif is_integer(eval_points):
            eval_points = pick_eval_points(
                ens=conn.pre, n_points=eval_points, rng=rng)
        else:
            eval_points = npext.array(eval_points, min_dims=2)

        x = np.dot(eval_points, encoders.T / conn.pre.radius)
        activities = model.dt * conn.pre.neurons.rates(x, gain, bias)
        if np.count_nonzero(activities) == 0:
            raise RuntimeError(
                "In '%s', for '%s', 'activites' matrix is all zero. "
                "This is because no evaluation points fall in the firing "
                "ranges of any neurons." % (str(conn), str(conn.pre)))

        if conn.function is None:
            targets = eval_points
        else:
            targets = np.zeros((len(eval_points), conn.function_size))
            for i, ep in enumerate(eval_points):
                targets[i] = conn.function(ep)

        if conn.weight_solver is not None:
            if conn.decoder_solver is not None:
                raise ValueError("Cannot specify both 'weight_solver' "
                                 "and 'decoder_solver'.")

            # account for transform
            targets = np.dot(targets, transform.T)
            transform = np.array(1., dtype=np.float64)

            decoders, solver_info = conn.weight_solver(
                activities, targets, rng=rng,
                E=model.params[conn.post].scaled_encoders.T)
            model.sig_out[conn] = model.sig_in[conn.post.neurons]
            signal_size = model.sig_out[conn].size
        else:
            solver = (conn.decoder_solver if conn.decoder_solver is
                      not None else nengo.decoders.lstsq_L2nz)
            decoders, solver_info = solver(activities, targets, rng=rng)
            signal_size = conn.dimensions

        # Add operator for decoders and filtering
        decoders = decoders.T
        if conn.synapse is not None and conn.synapse > model.dt:
            decay = decay_coef(pstc=conn.synapse, dt=model.dt)
            decoder_signal = Signal(
                decoders * (1.0 - decay),
                name="%s.decoders * (1 - decay)" % conn.label)
        else:
            decoder_signal = Signal(decoders,
                                    name="%s.decoders" % conn.label)
            decay = 0

        signal = Signal(np.zeros(signal_size), name=conn.label)
        model.operators.append(ProdUpdate(
            decoder_signal,
            model.sig_in[conn],
            Signal(decay, name="decay"),
            signal,
            tag="%s decoding" % conn.label))
    else:
        # Direct connection
        signal = model.sig_in[conn]

    # Add operator for filtering (in the case filter wasn't already
    # added, when pre.neurons is a non-direct Ensemble)
    if decoders is None and conn.synapse is not None:
        # Note: we add a filter here even if synapse < dt,
        # in order to avoid cycles in the op graph. If the filter
        # is explicitly set to None (e.g. for a passthrough node)
        # then cycles can still occur.
        signal = filtered_signal(signal, conn.synapse, model=model)

    if conn.modulatory:
        # Make a new signal, effectively detaching from post
        model.sig_out[conn] = Signal(
            np.zeros(model.sig_out[conn].size),
            name="%s.mod_output" % conn.label)
        # Add reset operator?
        # TODO: add unit test

    # Add operator for transform
    if isinstance(conn.post, nengo.objects.Neurons):
        if not model.has_built(conn.post):
            # Since it hasn't been built, it wasn't added to the Network,
            # which is most likely because the Neurons weren't associated
            # with an Ensemble.
            raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                               "that are not a part of any Ensemble." % (
                                   conn, conn.post))
        transform *= model.params[conn.post].gain[:, np.newaxis]

    model.operators.append(
        DotInc(Signal(transform, name="%s.transform" % conn.label),
               signal,
               model.sig_out[conn],
               tag=conn.label))

    # Set up probes
    for probe in conn.probes["signal"]:
        Builder.build(probe, dimensions=model.sig_out[conn].size, model=model)

    model.params[conn] = BuiltConnection(decoders=decoders,
                                         eval_points=eval_points,
                                         transform=transform,
                                         solver_info=solver_info)

Builder.register_builder(build_connection, nengo.Connection)  # noqa


def build_pyfunc(fn, t_in, n_in, n_out, label, model):
    if n_in:
        sig_in = Signal(np.zeros(n_in), name="%s.input" % label)
        model.operators.append(Reset(sig_in))
    else:
        sig_in = None

    if n_out > 0:
        sig_out = Signal(np.zeros(n_out), name="%s.output" % label)
    else:
        sig_out = None

    model.operators.append(
        SimPyFunc(output=sig_out, fn=fn, t_in=t_in, x=sig_in))

    return sig_in, sig_out


def build_direct(direct, dimensions, model):
    model.sig_in[direct] = Signal(np.zeros(dimensions), name=direct.label)
    model.sig_out[direct] = model.sig_in[direct]
    model.operators.append(Reset(model.sig_in[direct]))
    model.params[direct] = BuiltNeurons(gain=None, bias=None)

Builder.register_builder(build_direct, nengo.neurons.Direct)


def build_neurons(neurons, max_rates, intercepts, model):
    if neurons.n_neurons <= 0:
        raise ValueError(
            "Number of neurons (%d) must be positive." % neurons.n_neurons)
    gain, bias = neurons.gain_bias(max_rates, intercepts)
    model.sig_in[neurons] = Signal(
        np.zeros(neurons.n_neurons), name="%s.input" % neurons.label)
    model.sig_out[neurons] = Signal(
        np.zeros(neurons.n_neurons), name="%s.output" % neurons.label)

    model.operators.append(Copy(
        src=Signal(bias, name="%s.bias" % neurons.label),
        dst=model.sig_in[neurons]))

    for probe in neurons.probes["output"]:
        Builder.build(probe, dimensions=neurons.n_neurons, model=model)

    model.params[neurons] = BuiltNeurons(gain=gain, bias=bias)


def build_lifrate(lif, max_rates, intercepts, model):
    build_neurons(lif, max_rates, intercepts, model=model)
    model.operators.append(SimNeurons(
        neurons=lif, J=model.sig_in[lif], output=model.sig_out[lif]))

Builder.register_builder(build_lifrate, nengo.neurons.LIFRate)


def build_lif(lif, max_rates, intercepts, model):
    build_neurons(lif, max_rates, intercepts, model=model)
    voltage = Signal(np.zeros(lif.n_neurons), name="%s.voltage" % lif.label)
    refractory_time = Signal(
        np.zeros(lif.n_neurons), name="%s.refractory_time" % lif.label)
    model.operators.append(SimNeurons(neurons=lif,
                                      J=model.sig_in[lif],
                                      output=model.sig_out[lif],
                                      states=[voltage, refractory_time]))

Builder.register_builder(build_lif, nengo.neurons.LIF)


def build_alifrate(alif, max_rates, intercepts, model):
    build_neurons(alif, max_rates, intercepts, model=model)
    adaptation = Signal(np.zeros(alif.n_neurons),
                        name="%s.adaptation" % alif.label)
    model.operators.append(SimNeurons(neurons=alif,
                                      J=model.sig_in[alif],
                                      output=model.sig_out[alif],
                                      states=[adaptation]))

Builder.register_builder(build_alifrate, nengo.neurons.AdaptiveLIFRate)


def build_alif(alif, max_rates, intercepts, model):
    build_neurons(alif, max_rates, intercepts, model=model)
    voltage = Signal(np.zeros(alif.n_neurons), name="%s.voltage" % alif.label)
    refractory_time = Signal(np.zeros(alif.n_neurons),
                             name="%s.refractory_time" % alif.label)
    adaptation = Signal(np.zeros(alif.n_neurons),
                        name="%s.adaptation" % alif.label)
    model.operators.append(SimNeurons(
        neurons=alif,
        J=model.sig_in[alif],
        output=model.sig_out[alif],
        states=[voltage, refractory_time, adaptation]))

Builder.register_builder(build_alif, nengo.neurons.AdaptiveLIF)
