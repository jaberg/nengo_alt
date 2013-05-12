import pyopencl as cl
from ocl import array
from ocl.plan import Plan, Prog
from ocl.lif import plan_lif

class LIFMultiEnsemble(object):
    def __init__(self, n_populations, n_neurons_per, n_signals, signal_size,
            n_p2s=1,
            n_s2p=1,
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

    def _randomize_encoders(self, rng):
        pass

    def _randomize_decoders(self, rng):
        pass

    def prog(self, dt):

        # XXX add support for built-in filtering
        neuron_plan = plan_lif(self.queue,
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

        return Prog([neuron_plan])


