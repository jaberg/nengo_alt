try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy as np

import pyopencl as cl
import pyopencl.array

import theano
import theano.tensor as TT

from neuron import OCL_Neuron, Neuron
from ocl_util import CopySubRegion1D

class LIFNeuronView(object):
    def __init__(self, population, selection):
        self.population = population
        self.selection = selection

    def __len__(self):
        if isinstance(self.selection, slice):
            start, stop = self.selection.start, self.selection.stop 
            if start is None:
                start = 0
            if stop is None:
                stop = self.population.size
            return stop - start
        assert 0

    @property
    def start(self):
        return 0 if self.selection.start is None else self.selection.start

    @property
    def stop(self):
        if self.selection.stop is None:
            return self.population.size
        else:
            return self.selection.stop

    @property
    def step(self):
        return 1 if self.selection.step is None else self.selection.step



class OCL_LIFNeuron(OCL_Neuron):
    def __init__(self, queue, size, dt=0.001, tau_rc=0.02, tau_ref=0.002,
	upsample=2):
        """Constructor for a set of LIF rate neuron.

        :param int size: number of neurons in set
        :param float dt: timestep for neuron update function
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        OCL_Neuron.__init__(self, queue, size, dt)
        self.tau_rc = tau_rc
        self.tau_ref  = tau_ref
        self.V_threshold = 1.0
        self.voltage = cl.array.zeros(queue, (size,), 'float32')
        self.refractory_time = cl.array.zeros(queue, (size,), 'float32')
        self.input_current = cl.array.to_device(queue,
                5 * np.random.rand(size).astype('float32'))

	self.tau_rc_inv = 1.0 / tau_rc

        self.upsample = upsample
        self.upsample_dt = dt / upsample
	self.upsample_dt_inv = 1.0 / self.upsample_dt

        self._cl_fn = cl.Program(queue.context, """
            __kernel void foo(
                __global float *J,
                __global float *voltage,
                __global float *refractory_time,
                __global char *output
                         )
            {
                const float dt = %(upsample_dt)s;
                const float dt_inv = %(upsample_dt_inv)s;
                const float tau_ref = %(tau_ref)s;
                const float tau_rc_inv = %(tau_rc_inv)s;
                const float V_threshold = %(V_threshold)s;

                int gid = get_global_id(0);
                float v = voltage[gid];
                float rt = refractory_time[gid];
                float input_current = J[gid];
		char spiked = 0;

		for (int ii = 0; ii < %(upsample)s; ++ii)
                {
                  float dV = dt * tau_rc_inv * (input_current - v);
                  v += dV;
                  float post_ref = - rt * dt_inv;
                  v = v > 0 ?
                      v * (post_ref < 0 ? 0 : post_ref < 1 ? post_ref : 1)
                      : 0;
                  spiked |= v > V_threshold;
                  float overshoot = (v - V_threshold) / dV;
                  float spiketime = dt * (1.0 - overshoot);

                  v = v * (1.0 - spiked);
                  rt = spiked ? spiketime + tau_ref : rt - dt;
                }

                output[gid] = spiked;
                refractory_time[gid] = rt;
                voltage[gid] = v;
            }
            """ % self.__dict__).build().foo

    #TODO: make this generic so it can be applied to any neuron model
    # (by running the neurons and finding their response function),
    # rather than this special-case implementation for LIF

    def make_alpha_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        x1 = intercepts
        x2 = 1.0
        z1 = 1
        z2 = 1.0 / (1 - TT.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        alpha = (z1 - z2) / (x1 - x2)
        j_bias = z1 - alpha * x1
        return alpha, j_bias

    # TODO: have a reset() function at the ensemble and network level
    #that would actually call this
    def reset(self):
        """Resets the state of the neuron."""
        Neuron.reset(self)

        self.voltage.set_value(np.zeros(self.size).astype('float32'))
        self.refractory_time.set_value(np.zeros(self.size).astype('float32'))

    def cl_update(self, queue, dt):
        self._cl_fn(queue, (self.size,), None,
            self.input_current.data,
            self.voltage.data,
            self.refractory_time.data,
            self.output.data)

    def extend(self, Q, N):
        oldlen = len(self)
        cpy = CopySubRegion1D(Q.context, '=')
        def resize(arr):
            new_arr = cl.array.zeros(Q, oldlen + N, 'float32')
            cpy(Q, oldlen, arr.data, 0, new_arr.data, 0)
            Q.flush()
            return new_arr

        self.voltage = resize(self.voltage)
        self.refractory_time = resize(self.refractory_time)
        self.input_current = resize(self.input_current)
        self.output = resize(self.output)

    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, foo):
        return LIFNeuronView(self, foo)


class LIFNeuron(Neuron):
    def __init__(self, size, dt=0.001, tau_rc=0.02, tau_ref=0.002):
        """Constructor for a set of LIF rate neuron.

        :param int size: number of neurons in set
        :param float dt: timestep for neuron update function
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        Neuron.__init__(self, size, dt)
        self.tau_rc = tau_rc
        self.tau_ref  = tau_ref
        self.voltage = theano.shared(
            np.zeros(size).astype('float32'), name='lif.voltage')
        self.refractory_time = theano.shared(
            np.zeros(size).astype('float32'), name='lif.refractory_time')
        
    #TODO: make this generic so it can be applied to any neuron model
    # (by running the neurons and finding their response function),
    # rather than this special-case implementation for LIF        

    def make_alpha_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        x1 = intercepts
        x2 = 1.0
        z1 = 1
        z2 = 1.0 / (1 - TT.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        alpha = (z1 - z2) / (x1 - x2)
        j_bias = z1 - alpha * x1
        return alpha, j_bias

    # TODO: have a reset() function at the ensemble and network level
    #that would actually call this
    def reset(self):
        """Resets the state of the neuron."""
        Neuron.reset(self)

        self.voltage.set_value(np.zeros(self.size).astype('float32'))
        self.refractory_time.set_value(np.zeros(self.size).astype('float32'))

    def update(self, J):
        """Theano update rule that implementing LIF rate neuron type
        Returns dictionary with voltage levels, refractory periods,
        and instantaneous spike raster of neurons.

        :param float array J:
            the input current for the current time step

        """

        # Euler's method
        dV = self.dt / self.tau_rc * (J - self.voltage)

        # increase the voltage, ignore values below 0
        v = TT.maximum(self.voltage + dV, 0)  
        
        # handle refractory period        
        post_ref = 1.0 - (self.refractory_time - self.dt) / self.dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= TT.clip(post_ref, 0, 1)
        
        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked = TT.switch(v > 1, 1.0, 0.0)
        
        # adjust refractory time (neurons that spike get
        # a new refractory time set, all others get it reduced by dt)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV 
        spiketime = self.dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        new_refractory_time = TT.switch(
            spiked, spiketime + self.tau_ref, self.refractory_time - self.dt)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)
        # important that it's ordered, due to theano memory optimizations

        return OrderedDict({
                self.voltage: (v * (1 - spiked)).astype('float32'),
                self.refractory_time: new_refractory_time.astype('float32'),
                self.output: spiked.astype('float32'),
                })



