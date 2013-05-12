import pyopencl as cl
from ocl import array
from ocl.plan import Plan, Prog
from ocl.lif import plan_lif

class LIFMultiEnsemble(object):
    def __init__(self, n_populations, n_neurons_per, n_signals, signal_size,
            n_dec_per_signal=1,
            n_enc_per_population=1,
            lif_tau_rc=0.002,
            lif_tau_ref=0.002,
            lif_V_threshold=1.0,
            lif_upsample=2,
            potential_pstc=0.002, # ???
            noise=None,
            queue=None):
        self.__dict__.update(locals())
        del self.self

        # XXX Make Fortran layout an option (better for GPU)

        self.lif_v = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_rt = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_ic = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32')

        self.lif_output = array.zeros(queue,
                shape=(n_populations, n_neurons_per),
                dtype='float32') # XXX make int8

        self.signals = array.zeros(queue,
                shape=(n_signals, signal_size),
                dtype='float32')

        self.encoders = array.zeros(queue,
                shape=(signal_size, n_enc_per_population, n_populations),
                dtype='float32')

        self.encoders_signal_idx = array.zeros(queue,
                shape=(n_enc_per_population, n_populations),
                dtype='int32')

        self.decoders = array.zeros(queue,
                shape=(n_populations, signal_size, n_enc_per_population),
                dtype='float32')

        self.decoders_population_idx = array.zeros(queue,
                shape=(n_dec_per_signal, n_signals),
                dtype='int32')

    def _randomize_encoders(self, rng):
        encoders = rng.randn(*self.encoders.shape).astype('float32')
        self.encoders.set(encoders)

    def _randomize_decoders(self, rng):
        decoders = rng.randn(*self.decoders.shape).astype('float32')
        self.decoders.set(decoders)

    def neuron_plan(self, dt):
        # XXX add support for built-in filtering
        rval = plan_lif(self.queue,
                V=self.lif_v,
                RT=self.lif_rt,
                J=self.lif_ic,
                OV=self.lif_v,
                ORT=self.lif_rt,
                OS=self.lif_output,
                dt=dt,
                tau_rc=self.lif_tau_rc,
                tau_ref=self.lif_tau_ref,
                V_threshold=self.lif_V_threshold,
                upsample=self.lif_upsample,
                )
        return rval

    def decoder_plan(self):
        raise NotImplementedError()

    def encoder_plan(self):
        raise NotImplementedError()

    def prog(self, dt):
        neuron_plan = self.neuron_plan(dt)

        return Prog([neuron_plan])


