import numpy as np
import pyopencl as cl

class CopySubRegion1D(object):
    def __init__(self, context, operation = '='):
        self.fn = cl.Program(context, """
        __kernel void fn(
            __global const float *A_data,
            const int A_offset,
            __global float *Y_data,
            const int Y_offset
                         )
        {
            const int bb = get_global_id(0);
            Y_data[Y_offset + bb] %(operation)s A_data[A_offset + bb];
        }
        """ % locals()).build().fn

    def __call__(self, queue, N, A, Aoffset, B, Boffset):
        self.fn(queue, (N,), None, A, np.intc(Aoffset), B, np.intc(Boffset))

