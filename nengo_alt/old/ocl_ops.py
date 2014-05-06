import re
import numpy as np
import theano
import theano.tensor
import pyopencl as cl

import simulator
from simulator_ocl import (
    alloc,
    perform,
    ifs_arrays,
    ifs_consts,
    ifs_set_arrays,
    ifs_set_consts,
    UnAllocatedOutput,
    )
from ocl.array import Array, to_device, empty
from ocl.gemv_batched import plan_map_gemv
from ocl.gemv_batched import plan_misc_gemv
from ocl.dot import plan_dot
from ocl.plan import Plan
from ocl.elemwise import plan_elemwise
import lif
import probe

@alloc(theano.tensor.basic.Alloc)
def alloc_a(queue, ifs, node):
    # -- set up a view of X
    # XXX make sure that dimshuffle is inplace
    in_vars = ifs_arrays(ifs, node.inputs)
    in_consts = ifs_consts(ifs, node.inputs)
    shape = in_consts[1:]
    if in_vars[0] is UnAllocatedOutput:
        val = in_consts[0]
        Y = np.empty(shape, val.dtype)
        Y[...] = val
        ifs_set_consts(ifs, node.outputs, [Y])
    else:
        val = in_vars[0]
        Y = empty(queue, shape, val.dtype)
        ifs_set_arrays(ifs, node.outputs, [Y])


@perform(theano.tensor.basic.Alloc)
def alloc_p(queue, ifs, node):
    in_vars = ifs_arrays(ifs, node.inputs)
    in_consts = ifs_consts(ifs, node.inputs)
    Y, = ifs_arrays(ifs, node.outputs)
    if in_vars[0] is UnAllocatedOutput:
        return
        Ytype = Y.ocldtype
        Ys0, Ys1, Ys2 = Y.itemstrides + (1,) * (3 - Y.ndim)
        Xval = in_consts[0]
        _fn = cl.Program(queue.context, """
            __kernel void foo(
                __global %(Ytype)s *Y)
            {
                int gid0 = get_global_id(0);
                int gid1 = get_global_id(1);
                int gid2 = get_global_id(2);
                Y[gid0 * Ys0 + gid1 * Ys1 + gid2 * Ys2] = %(Xval)s;
            }
            """ % locals()).build().foo
        _fn_args = (queue, (Y.size,), None, Y.data)
        return [Plan(locals())]
    else:
        raise NotImplementedError()


@alloc(theano.compile.ops.DeepCopyOp)
def deep_copy_a(queue, ifs, node):
    X, = ifs_arrays(ifs, node.inputs)
    Xc, = ifs_consts(ifs, node.inputs)
    if Xc is UnAllocatedOutput:
        Y = X.empty_like()
        ifs_set_arrays(ifs, node.outputs, [Y])
    else:
        ifs_set_consts(ifs, node.inputs, [Xc])


@perform(theano.compile.ops.DeepCopyOp)
def deep_copy_p(queue, ifs, node):
    X, = ifs_arrays(ifs, node.inputs)
    Xc, = ifs_consts(ifs, node.inputs)
    if Xc is UnAllocatedOutput:
        Y, = ifs_arrays(ifs, node.outputs)
        raise NotImplementedError()


@alloc(theano.tensor.elemwise.DimShuffle)
def dimshuffle_a(queue, ifs, node):
    # -- set up a view of X
    # XXX make sure that dimshuffle is inplace
    Xvar, = node.inputs
    X = ifs.meta[Xvar].ocl0
    if X is UnAllocatedOutput:
        Xval = ifs.meta[Xvar].const_val
        outputs = [[None]]
        node.op.perform(node, [Xval], outputs)
        ifs_set_consts(ifs, node.outputs, outputs[0][0])
    else:
        Yvar, = node.outputs

        Yshape = list(X.shape)
        Ystrides = list(X.strides)

        # -- drop
        for drop in reversed(node.op.drop):
            Yshape.pop(drop)
            Ystrides.pop(drop)

        # -- transpose
        Yshape = [Yshape[i] for i in node.op.shuffle]
        Ystrides = [Ystrides[i] for i in node.op.shuffle]

        # -- augment
        for augm in node.op.augment:
            Yshape.insert(augm, 1)
            Ystrides.insert(augm, X.dtype.itemsize)

        Y = Array(queue, data=X.data, dtype=X.dtype,
                  shape=Yshape, strides=Ystrides)
        ifs.meta[Yvar].ocl0 = Y


@perform(theano.tensor.elemwise.DimShuffle)
def dimshuffle_p(queue, ifs, node):
    return []


@alloc(theano.tensor.basic.IncSubtensor)
def inc_subtensor_a(queue, ifs, node):
    # -- set up a view of X
    # XXX make sure that dimshuffle is inplace
    in_vars = ifs_arrays(ifs, node.inputs)
    in_consts = ifs_consts(ifs, node.inputs)
    if len(in_vars) > 2:
        raise NotImplementedError()
    if in_vars[0] is UnAllocatedOutput:
        if in_vars[1] is UnAllocatedOutput:
            # -- two constants -> constant output
            Y = in_consts[0].copy()
            Y.__setitem__(node.op.idx_list, in_consts[1])
            ifs_set_consts(ifs, node.outputs, [Y])
        else:
            Y = to_device(queue, in_consts[0])
            # -- works because len(in_vars) <= 2 guarantees const idx
            Xslice = in_consts[0].__getitem__(*node.op.idx_list)
            Xorig = to_device(queue, Xslice)
            ifs.meta[node.outputs[0]].Xorig = Xorig
            ifs_set_arrays(ifs, node.outputs, [Y])
    else:
        Y = in_vars[0].empty_like()
        ifs_set_arrays(ifs, node.outputs, [Y])

#@back_alloc(theano.tensor.basic.IncSubtensor)
def inc_subtensor_ba(queue, ifs, node):
    Y, = ifs_arrays(ifs, node.outputs)
    X, A = ifs_arrays(ifs, node.inputs)
    if node.op.set_instead_of_inc:
        if A is not UnAllocatedOutput:
            YA = Y.__getitem__(node.op.idx_list)
            ifs.meta[node.inputs[1]].ocl0 = YA

@perform(theano.tensor.basic.IncSubtensor)
def inc_subtensor_p(queue, ifs, node):
    X, A = ifs_arrays(ifs, node.inputs)
    Xc, Ac = ifs_consts(ifs, node.inputs)
    Y, = ifs_arrays(ifs, node.outputs)
    YA = Y.__getitem__(node.op.idx_list)
    if node.op.set_instead_of_inc:
        if A.same_view_as(YA):
            return []
        body = "$OUT_0 = $IN_0;"
        if X is UnAllocatedOutput:
            return plan_elemwise(queue, body, [A], [YA])
        else:
            return plan_elemwise(queue, body, [A], [YA])
    else:
        body = "$OUT_0 = $IN_0 + $IN_1;"
        if X is UnAllocatedOutput:
            return plan_elemwise(queue, body, 
                    [ifs.meta[node.outputs[0]].Xorig, A], [YA])
        else:
            return plan_elemwise(queue, body, 
                    [X.__getitem__(node.op.idx_list), A], [YA])


@alloc(theano.tensor.opt.MakeVector)
def make_vector_a(queue, ifs, node):
    inputs = ifs_consts(ifs, node.inputs)
    if UnAllocatedOutput in inputs:
        theano.printing.debugprint(node.outputs)
        raise NotImplementedError('non-constant MakeVector')
    ifs_set_consts(ifs, node.outputs, [np.asarray(inputs)])


@perform(theano.tensor.opt.MakeVector)
def make_vector_p(queue, sim, node):
    return []


@alloc(simulator.MapGemv)
def map_gemv_a(queue, ifs, node):
    Y = ifs_arrays(ifs, node.inputs)[-1]
    if Y is UnAllocatedOutput:
        Y = ifs_consts(ifs, node.inputs)[-1]
        ifs_set_arrays(ifs, node.outputs, [empty(queue, Y.shape, Y.dtype)])
    else:
        if node.destroy_map:
            ifs_set_arrays(ifs, node.outputs, [Y])
        else:
            ifs_set_arrays(ifs, node.outputs, [empty(queue, Y.shape, Y.dtype)])


@perform(simulator.MapGemv)
def map_gemv_p(queue, ifs, node):
    alpha, A, X, beta, Y_in = ifs_arrays(ifs, node.inputs)
    Y_out, = ifs_arrays(ifs, node.outputs)

    # XXX: following depends on constants alpha, beta
    falpha = float(node.inputs[0].data)
    fbeta = float(node.inputs[3].data)

    B, M, N = A.shape
    Bx, Nx = X.shape
    By, My = Y_out.shape
    assert Bx == By == B
    assert My == M
    assert Nx == N

    #A_offsets = to_device(queue, np.arange(B) * M * N )
    #X_offsets = to_device(queue, np.arange(B) * N )
    #Y_offsets = to_device(queue, np.arange(B) * M)

    if Y_in is UnAllocatedOutput:
        Y_in = None
        Y_in_val = ifs.meta[node.inputs[-1]].const_val
        if np.all(Y_in_val == 0):
            fbeta = 0
        elif fbeta != 0:
            Y_in = to_device(queue, np.asarray(Y_in_val * fbeta))
            fbeta = 1

    return [plan_map_gemv(queue, falpha, A, X, fbeta, Y_out, Y_in)]


@alloc(simulator.MiscGemv)
def misc_gemv_a(queue, ifs, node):
    return map_gemv_a(queue, ifs, node)


@perform(simulator.MiscGemv)
def misc_gemv_p(queue, ifs, node):
    alpha, A, X, Xi, beta, Y_in = ifs_arrays(ifs, node.inputs)
    Y_out, = ifs_arrays(ifs, node.outputs)

    # XXX: following depends on constants alpha, beta
    falpha = float(node.inputs[0].data)
    fbeta = float(node.inputs[4].data)

    if Xi is UnAllocatedOutput:
        Xi = to_device(queue, ifs.meta[node.inputs[3]].const_val)

    if A is UnAllocatedOutput:
        A = to_device(queue, ifs.meta[node.inputs[1]].const_val)
    
    B, M, N = A.shape
    Bx, Nx = X.shape
    if N != Nx:
        theano.printing.debugprint(node.inputs[1])
        theano.printing.debugprint(node.inputs[2])
        raise ValueError('shape mismatch', (A.shape, X.shape))
    By, My = Y_out.shape
    Bxi, = Xi.shape
    assert Bxi == By == B
    assert My == M

    #A_offsets = to_device(queue, np.arange(B) * M * N )
    #X_offsets = to_device(queue, np.arange(B) * N )
    #Y_offsets = to_device(queue, np.arange(B) * M)

    if Y_in is UnAllocatedOutput:
        Y_in = None
        Y_in_val = ifs.meta[node.inputs[-1]].const_val
        if np.all(Y_in_val == 0):
            fbeta = 0
        elif fbeta != 0:
            Y_in = to_device(queue, np.asarray(Y_in_val * fbeta))
            fbeta = 1

    return [plan_misc_gemv(queue, falpha, A, X, Xi, fbeta, Y_out, Y_in)]


@alloc(theano.tensor.basic.Reshape)
def reshape_a(queue, ifs, node):
    X, shp = node.inputs
    Xval = ifs.meta[X].ocl0
    shape_val = ifs.meta[shp].const_val
    if shape_val is UnAllocatedOutput:
        theano.printing.debugprint(shp)
        raise NotImplementedError('need constant shape', shp)
    try:
        shape_val = [int(shape_val)]
    except:
        pass

    assert len(shape_val) == node.outputs[0].ndim
    assert node.outputs[0].dtype == node.inputs[0].dtype

    if np.prod(Xval.shape) == np.prod(shape_val) == 1:
        Yval = Array(queue, data=Xval.data, dtype=Xval.dtype,
                shape=list(shape_val),
                strides=[Xval.dtype.itemsize] * len(shape_val))
    else:
        theano.printing.debugprint(node.outputs)
        print 'X stats', Xval.shape, Xval.strides
        print 'target shape', shape_val
        raise NotImplementedError('MakeVector')
    ifs_set_arrays(ifs, node.outputs, [Yval])

@perform(theano.tensor.basic.Reshape)
def reshape_p(queue, sim, node):
    return []


@alloc(theano.tensor.opt.Shape_i)
def shape_i_a(queue, ifs, node):
    X, = ifs_arrays(ifs, node.inputs)
    ifs_set_consts(ifs, node.outputs, [X.shape[node.op.i]])


@perform(theano.tensor.opt.Shape_i)
def shape_i_p(queue, sim, node):
    return []


@alloc(theano.tensor.basic.Subtensor)
def subtensor_a(queue, ifs, node):
    # -- set up a view of X
    # XXX make sure that dimshuffle is inplace
    in_vars = ifs_arrays(ifs, node.inputs)
    in_consts = ifs_consts(ifs, node.inputs)
    if len(in_vars) == 1:
        X, = in_vars
        if 0 in node.op.view_map:
            Y = X.__getitem__(node.op.idx_list)
            ifs_set_arrays(ifs, node.outputs, [Y])
        else:
            raise NotImplementedError(node.op.idx_list)
    else:
        raise NotImplementedError(node.op.idx_list)


@perform(theano.tensor.basic.Subtensor)
def subtensor_p(queue, sim, node):
    return []


def flatten_c_contig(Xval):
    # currently this is a little different from the c contiguous
    # flag logic in array.Array, so we redo it here
    need_stride = Xval.dtype.itemsize
    c_contig = True
    for si, ri in reversed(zip(Xval.shape, Xval.strides)):
        if si == 1:
            continue
        else:
            if ri == need_stride:
                need_stride *= si
            else:
                c_contig = False
    return c_contig

@alloc(theano.tensor.basic.Flatten)
def flatten_a(queue, ifs, node):
    X,= node.inputs
    Xval = ifs.meta[X].ocl0
    if flatten_c_contig(Xval):
        Yval = Array(queue, data=Xval.data, dtype=Xval.dtype,
                shape=[int(np.prod(Xval.shape))],
                strides=[Xval.dtype.itemsize])
    else:
        raise NotImplementedError()
    ifs.meta[node.outputs[0]].ocl0 = Yval

@perform(theano.tensor.basic.Flatten)
def flatten_p(queue, ifs, node):
    X, = node.inputs
    Xval, = ifs_arrays(ifs, node.inputs)
    Y, = node.outputs
    Yval, = ifs_arrays(ifs, node.outputs)
    if Xval.data is Yval.data:
        return []
    elif flatten_c_contig(Xval) and flatten_c_contig(Yval):
        Xtype = Xval.ocldtype
        Ytype = Yval.ocldtype
        _fn = cl.Program(queue.context, """
            __kernel void foo(
                __global const %(Xtype)s *X,
                __global %(Ytype)s *Y)
            {
                int ii = get_global_id(0);
                Y[ii] = X[ii];
            }
            """ % locals()).build().foo
        _fn_args = (queue, (Xval.size,), None, Xval.data, Yval.data)
        return [Plan(locals())]

    else:
        raise NotImplementedError('Flatten', (Xval.shape, Yval.shape))


@alloc(theano.tensor.basic.Dot)
def dot_a(queue, ifs, node):
    Xv, Yv = ifs_arrays(ifs, node.inputs)
    Xc, Yc = ifs_consts(ifs, node.inputs)

    # TODO: refactor to make this more general
    if Yc is not UnAllocatedOutput:
        # Y is constant
        raise NotImplementedError()
    else:
        # Y is not constant
        if Xc is not UnAllocatedOutput:
            # X is constant
            if Xc.ndim == 2 and Yv.ndim == 2:
                Zshape = Yv.shape
                Zdtype = np.dtype(node.outputs[0].dtype)
                Zstrides = [Yv.shape[1] * Zdtype.itemsize,
                            Zdtype.itemsize]
            else:
                raise NotImplementedError()
        else:
            # X is not constant
            if Xc.ndim == 2 and Yv.ndim == 2:
                Zshape = [Xv.shape[0], Yv.shape[1]]
                Zdtype = np.dtype(node.outputs[0].dtype)
                Zstrides = [Yv.shape[1] * Zdtype.itemsize,
                            Zdtype.itemsize]
            else:
                raise NotImplementedError()

    size = Zstrides[0] * Zshape[0]
    Zdata = cl.Buffer(queue.context,
                      flags=cl.mem_flags.READ_WRITE,
                      size=int(size))
    Zval = Array(queue, data=Zdata, dtype=Zdtype,
                 shape=Zshape, strides=Zstrides)
    ifs_set_arrays(ifs, node.outputs, [Zval])

@perform(theano.tensor.basic.Dot)
def dot_p(queue, ifs, node):
    Xv, Yv = ifs_arrays(ifs, node.inputs)
    Xc, Yc = ifs_consts(ifs, node.inputs)
    Zv, = ifs_arrays(ifs, node.outputs)

    # TODO: refactor to make this more general
    if Yc is not UnAllocatedOutput:
        # Y is constant
        raise NotImplementedError()
    else:
        # Y is not constant
        if Xc is not UnAllocatedOutput:
            # X is constant
            if Xc.ndim == 2 and Yv.ndim == 2:
                if Xc.size != 1:
                    raise NotImplementedError()
                X = float(Xc)

                assert Yv.shape[0] == 1

                Ys0, Ys1 = Yv.itemstrides
                Zs0, Zs1 = Zv.itemstrides
                Ytype = Yv.ocldtype
                Ztype = Zv.ocldtype
                sumtype = Ztype # TODO: consider more precision here

                _fn = cl.Program(queue.context, """
                    __kernel void foo(
                        __global const %(Ytype)s *Y,
                        __global %(Ztype)s *Z)
                    {
                        int ii = get_global_id(0);
                        int jj = get_global_id(1);
                        Z[ii * %(Zs0)s + jj * %(Zs1)s] =
                            %(X)s * Y[ii * %(Ys0)s + jj * %(Ys1)s];
                    }
                    """ % locals()).build().foo

                _fn_args = (queue, Zv.shape, None, Yv.data, Zv.data)
                return [Plan(locals())]
            else:
                raise NotImplementedError()
        else:
            # X is not constant
            return plan_dot(queue, Xv, Yv, Zv)

@alloc(lif.LIF_Op)
def lif_a(queue, ifs, node):
    v_v, rt_v, ic_v, dt_v = ifs_arrays(ifs, node.inputs)
    v_c, rt_c, ic_c, dt_c = ifs_consts(ifs, node.inputs)
    ifs_set_arrays(ifs, node.outputs,
                   [v_v.empty_like(),
                    rt_v.empty_like(),
                    empty(queue, v_v.shape, dtype=np.float32)])


@perform(lif.LIF_Op)
def lif_p(queue, ifs, node):
    V, RT, J, dt_v = ifs_arrays(ifs, node.inputs)
    v_c, rt_c, ic_c, dt_c = ifs_consts(ifs, node.inputs)
    OV, ORT, OS = ifs_arrays(ifs, node.outputs)

    dt = float(dt_c)
    
    tau_rc = node.op.tau_rc
    tau_ref  = node.op.tau_ref
    V_threshold = 1.0
    tau_rc_inv = 1.0 / tau_rc

    upsample = node.op.upsample
    upsample_dt = dt / upsample
    upsample_dt_inv = 1.0 / upsample_dt

    Jtype = J.ocldtype
    Vtype = V.ocldtype
    RTtype = RT.ocldtype
    OStype = OS.ocldtype

    _fn = cl.Program(queue.context, """
        __kernel void foo(
            __global const %(Jtype)s *J,
            __global const %(Vtype)s *voltage,
            __global const %(RTtype)s *refractory_time,
            __global %(Vtype)s *out_voltage,
            __global %(RTtype)s *out_refractory_time,
            __global %(OStype)s *out_spiked
                     )
        {
            const %(RTtype)s dt = %(upsample_dt)s;
            const %(RTtype)s dt_inv = %(upsample_dt_inv)s;
            const %(RTtype)s tau_ref = %(tau_ref)s;
            const %(Vtype)s tau_rc_inv = %(tau_rc_inv)s;
            const %(Vtype)s V_threshold = %(V_threshold)s;

            const int gid = get_global_id(0);
            %(Vtype)s v = voltage[gid];
            %(RTtype)s rt = refractory_time[gid];
            %(Jtype)s input_current = J[gid];
            int spiked = 0;

            for (int ii = 0; ii < %(upsample)s; ++ii)
            {
              %(Vtype)s dV = dt * tau_rc_inv * (input_current - v);
              %(RTtype)s post_ref = 1.0 - (rt - dt) * dt_inv;
              v += dV;
              v = v > 0 ?
                  v * (post_ref < 0 ? 0.0 : post_ref < 1 ? post_ref : 1.0)
                  : 0;
              const int spiked_ii = v > V_threshold;
              %(Vtype)s overshoot = (v - V_threshold) / dV;
              %(RTtype)s spiketime = dt * (1.0 - overshoot);

              if (spiked_ii)
              {
                v = 0.0;
                rt = spiketime + tau_ref;
                spiked = 1;
              }
              else
              {
                rt -= dt;
              }
            }

            out_voltage[gid] = v;
            out_refractory_time[gid] = rt;
            out_spiked[gid] = spiked ? (%(OStype)s) 1 : (%(OStype)s) 0;
        }
        """ % locals()).build().foo

    # XXX ASSERT C CONTIGUOUS

    _fn_args = (queue, (V.size,), None,
                J.data, V.data, RT.data,
                OV.data, ORT.data, OS.data)
    return [Plan(locals())]

def concat_lif_populations(ifs):
    for node in ifs.fg.toposort():
        if isinstance(node.op, lif.LIF_Op):
            print node.op
            for vv in node.inputs:
                print vv, dir(ifs.meta[vv])


def concat_connections(ifs):
    pass


@alloc(theano.tensor.elemwise.Elemwise)
def elemwise_a(queue, ifs, node):
    ocl_inputs = ifs_arrays(ifs, node.inputs)
    const_inputs = ifs_consts(ifs, node.inputs)
    ocl_outputs = []
    for outnum, vv in enumerate(node.outputs):
        if outnum in node.op.destroy_map:
            destroyed_in, = node.op.destroy_map[outnum]
            ocl_outputs.append(ocl_inputs[destroyed_in])
        else:
            shape = np.asarray([1] * vv.ndim)
            for vi in ocl_inputs:
                if vi is not UnAllocatedOutput:
                    assert len(shape) == len(vi.shape), (shape, vi.shape)
                    shape = list(np.maximum(shape, vi.shape))
            for vi in const_inputs:
                if vi is not UnAllocatedOutput and hasattr(vi, 'shape'):
                    assert len(shape) == len(vi.shape)
                    shape = list(np.maximum(shape, vi.shape))
            ocl_outputs.append(empty(queue, shape, np.dtype(vv.dtype)))
    ifs_set_arrays(ifs, node.outputs, ocl_outputs)


@perform(theano.tensor.elemwise.Elemwise)
def elemwise_p(queue, ifs, node):
    ocl_inputs = ifs_arrays(ifs, node.inputs)
    const_inputs = ifs_consts(ifs, node.inputs)
    ocl_outputs = ifs_arrays(ifs, node.outputs)

    # generate a loop body suitable for plan_elemwise
    # plan_elemwise requires that inputs be called
    # $IN_<i> for i in range(len(inputs))
    # $OUT_<i> for i in range(len(outputs))
    #
    # plan_elemwise will not see the constant inputs
    # to the theano node, so we fold them into the loop body here

    c_body_inputs = {}
    n_in_vars = 0
    for vv, ic in zip(node.inputs, const_inputs):
        if ic is UnAllocatedOutput:
            c_body_inputs[vv] = '$IN_%i' % n_in_vars
            n_in_vars += 1
        else:
            # -- XXX float covers ints too... mostly
            c_body_inputs[vv] = str(float(ic))
    for inum, vv in enumerate(node.outputs):
        c_body_inputs[vv] = '$OUT_%i' % inum

    scalar_inputs = [theano.scalar.Scalar(dtype=vv.dtype)()
                     for vv in node.inputs]
    ctype_body = node.op.scalar_op.c_code(
        node.op.scalar_op(*scalar_inputs).owner,
        'name',
        [c_body_inputs[vv] for vv in node.inputs],
        [c_body_inputs[vv] for vv in node.outputs],
        {})

    # -- replace the numpy typedefs
    ctype_body = re.sub('npy_float64', 'double', ctype_body)
    ctype_body = re.sub('npy_float32', 'float', ctype_body)
    if 'npy' in ctype_body:
        raise NotImplementedError()

    return plan_elemwise(queue, ctype_body,
         [v for v in ocl_inputs if v is not UnAllocatedOutput],
         ocl_outputs)


@alloc(theano.tensor.Rebroadcast)
def rebroadcast_a(queue, ifs, node):
    ifs_set_arrays(ifs, node.outputs, ifs_arrays(ifs, node.inputs))
    ifs_set_consts(ifs, node.outputs, ifs_consts(ifs, node.inputs))


@perform(theano.tensor.Rebroadcast)
def rebroadcast_p(queue, sim, node):
    return []


@alloc(probe.Scribe)
def scribe_a(queue, ifs, node):
    x, buf, i, t, dt_sample = ifs_arrays(ifs, node.inputs)
    ifs_set_arrays(ifs, node.outputs, [buf, i.empty_like()])


@perform(probe.Scribe)
def scribe_p(queue, ifs, node):
    # Scribes are handled specially by the simulator
    # because generally, the simulator does not permit the size of Array
    # objects to change during simulation.
    # Scribe ops are a necessary violation of that general rule.

    X, buf, i, t, dt_sample = ifs_arrays(ifs, node.inputs)
    obuf, oi, = ifs_arrays(ifs, node.outputs)

    # dt_sample must be constant
    dt_sample = float(ifs_consts(ifs, node.inputs)[-1])

    def ctype(obj):
        return cl.tools.dtype_to_ctype(obj.dtype)

    Xtype = ctype(X)
    itype = ctype(i)
    ttype = ctype(t)
    btype = ctype(buf)

    if buf.ndim == 2:
        Xs0, = X.itemstrides
        obuf_s0, obuf_s1 = obuf.itemstrides
        text = """
            __kernel void foo(
                __global const %(Xtype)s * X,
                __global const %(itype)s * i,
                __global const %(ttype)s * t,
                __global %(btype)s * obuf,
                __global %(itype)s * oi
                         )
            {
                const int gid0 = get_global_id(0);
                int i_samp = (t[0] / %(dt_sample)s);
                int i0 = i[0];
                if (i_samp > i0)
                {
                    obuf[i_samp * %(obuf_s0)s + gid0 * %(obuf_s1)s]
                        = X[gid0 * %(Xs0)s];
                    oi[0] = i_samp;
                }
                else
                {
                    oi[0] = i0;
                }
            }
            """ % locals()
    elif buf.ndim == 3:
        Xs0, Xs1 = X.itemstrides
        obuf_s0, obuf_s1, obuf_s2 = obuf.itemstrides
        text = """
            __kernel void foo(
                __global const %(Xtype)s * X,
                __global const %(itype)s * i,
                __global const %(ttype)s * t,
                __global %(btype)s * obuf,
                __global %(itype)s * oi
                         )
            {
                const int gid0 = get_global_id(0);
                const int gid1 = get_global_id(1);

                int i_samp = (t[0] / %(dt_sample)s);
                int i0 = i[0];
                if (i_samp > i0)
                {
                    obuf[i_samp * %(obuf_s0)s
                            + gid0 * %(obuf_s1)s
                            + gid1 * %(obuf_s2)s]
                        = X[gid0 * %(Xs0)s + gid1 * %(Xs1)s];
                    oi[0] = i_samp;
                }
                else
                {
                    oi[0] = i0;
                }
            }
            """ % locals()
    else:
        raise NotImplementedError('buf ndim', buf.ndim)

    _fn = cl.Program(queue.context, text).build().foo
    _fn_args = (queue, X.shape, None,
        X.data, i.data, t.data,
        obuf.data, oi.data)
    return [Plan(locals())]

if 0:

        # -- set up the outputs from plan 0 as the inputs for plan 1
        # -- and the outputs from plan 1 as the inputs for plan 0
        for (ivar, ovar) in self.step.fn.updated_vars.items():
            if ovar not in self._ocl_vars[0]:
                # -- this can happen if `ovar` is not
                #    implicated in any computation except
                #    the updating of this variable
                #    and perhaps being updated itself.
                assert ovar.owner is None
                if hasattr(vv, 'data'):
                    self.constant_vars[ovar] = ovar.data
                    raise NotImplementedError('copy const into ocl ivar')
                else:
                    val = ovar.get_value(borrow=True)
                    self._ocl_vars[0][ovar] = to_device(self.queue, val)
                    self.queue.finish()
            self._ocl_vars[1][ivar] = self._ocl_vars[0][ovar]
            if ivar not in self._ocl_vars[0]:
                # -- if ivar is not an input to anything, then this is the
                #    first time we've seen it
                val = ivar.get_value(borrow=True)
                self._ocl_vars[0][ivar] = to_device(self.queue, val)
                self.queue.finish()

            self._ocl_vars[1][ovar] = self._ocl_vars[0][ivar]

        # -- allocate workspace for the second plan (plan 1)
        self.ocl_vars = self._ocl_vars[1]
        for node in self.ifs.nodes:
            for vv in node.inputs:
                if vv in self._ocl_vars[1]:
                    continue
                if vv in self.constant_vars:
                    continue
                assert not hasattr(vv, 'data')
                if vv.name == 'simulation_time':
                    self._ocl_vars[1][vv] = self._simtime[1]
                elif vv.owner is None:
                    # -- vv is a shared var that isn't updated
                    self._ocl_vars[1][vv] = self._ocl_vars[0][vv]
            if any(vv not in self.ocl_vars for vv in node.outputs):
                ocl_alloc[type(node.op)](self.queue, self, node)
                for vout in node.outputs:
                    if vout in self._ocl_vars[1]:
                        assert self._ocl_vars[1][vout].ndim == vout.ndim, node.op
                        assert self._ocl_vars[1][vout].dtype == vout.dtype, node.op
                    else:
                        assert vout in self.constant_vars
        del self.ocl_vars


        # -- build plans for evaluating ocl_vals[0]
        for node in self.ifs.nodes:
            self.ocl_vars = self._ocl_vars[1]
            self.ocl_vars = self._ocl_vars[1]
            plans = ocl_perform[type(node.op)](self.queue, self, node)
            for plan in plans:
                plan.node = node
            self._plans[1].extend(plans)
            self._node_plans[1][node] = plans
        del self.ocl_vars
        self.queue.finish()
