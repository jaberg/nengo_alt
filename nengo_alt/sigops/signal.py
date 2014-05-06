import numpy as np


class ShapeMismatch(ValueError):
    pass


class SignalView(object):

    def __init__(self, base, shape, elemstrides, offset, name=None):
        assert base is not None
        self.base = base
        self.shape = tuple(shape)
        self.elemstrides = tuple(elemstrides)
        self.offset = int(offset)
        if name is not None:
            self._name = name

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    def __repr__(self):
        return '%s{%s, %s}' % (
            self.__class__.__name__,
            self.name, self.shape)

    def view_like_self_of(self, newbase, name=None):
        if newbase.base != newbase:
            raise NotImplementedError()
        if newbase.structure != self.base.structure:
            raise NotImplementedError('technically ok but should not happen',
                                      (self.base, newbase))
        return SignalView(newbase,
                          self.shape,
                          self.elemstrides,
                          self.offset,
                          name)

    @property
    def structure(self):
        return (self.shape, self.elemstrides, self.offset)

    def same_view_as(self, other):
        return self.structure == other.structure and self.base == other.base

    @property
    def dtype(self):
        return np.dtype(self.base.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        if self.elemstrides == (1,):
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1]
            for si in reversed(shape[1:]):
                elemstrides = [si * elemstrides[0]] + elemstrides
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        elif self.size == 1:
            # -- scalars can be reshaped to any number of (1, 1, 1...)
            size = int(np.prod(shape))
            if size != self.size:
                raise ShapeMismatch(shape, self.shape)
            elemstrides = [1] * len(shape)
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=self.offset)
        else:
            # -- there are cases where reshaping can still work
            #    but there are limits too, because we can only
            #    support view-based reshapes. So the strides have
            #    to work.
            raise NotImplementedError('reshape of strided view')

    def transpose(self, neworder=None):
        if neworder:
            raise NotImplementedError()
        return SignalView(
            self.base,
            reversed(self.shape),
            reversed(self.elemstrides),
            self.offset,
            self.name + '.T'
        )

    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return self.transpose()

    def __getitem__(self, item):  # noqa
        # -- copy the shape and strides
        shape = list(self.shape)
        elemstrides = list(self.elemstrides)
        offset = self.offset
        if isinstance(item, (list, tuple)):
            dims_to_del = []
            for ii, idx in enumerate(item):
                if isinstance(idx, int):
                    dims_to_del.append(ii)
                    offset += idx * elemstrides[ii]
                elif isinstance(idx, slice):
                    start, stop, stride = idx.indices(shape[ii])
                    offset += start * elemstrides[ii]
                    if stride != 1:
                        raise NotImplementedError()
                    shape[ii] = stop - start
            for dim in reversed(dims_to_del):
                shape.pop(dim)
                elemstrides.pop(dim)
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, (int, np.integer)):
            if len(self.shape) == 0:
                raise IndexError()
            if not (0 <= item < self.shape[0]):
                raise NotImplementedError()
            shape = self.shape[1:]
            elemstrides = self.elemstrides[1:]
            offset = self.offset + item * self.elemstrides[0]
            return SignalView(
                base=self.base,
                shape=shape,
                elemstrides=elemstrides,
                offset=offset)
        elif isinstance(item, slice):
            return self.__getitem__((item,))
        else:
            raise NotImplementedError(item)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            if self.base is self:
                return '<anon%d>' % id(self)
            else:
                return 'View(%s[%d])' % (self.base.name, self.offset)

    @name.setter
    def name(self, value):
        self._name = value

    def is_contiguous(self):
        shape, strides, offset = self.structure
        if len(shape) == 0:
            return True, offset, offset + 1
        elif len(shape) == 1:
            if strides[0] == 1:
                return True, offset, offset + shape[0]
            else:
                return False, None, None
        elif len(shape) == 2:
            if strides == (1, shape[0]) or strides == (shape[1], 1):
                return True, offset, offset + shape[0] * shape[1]
            else:
                return False, None, None
        else:
            raise NotImplementedError()
        # if self.ndim == 1 and self.elemstrides[0] == 1:
            # return self.offset, self.offset + self.size

    def shares_memory_with(self, other):  # noqa
        # TODO: WRITE SOME UNIT TESTS FOR THIS FUNCTION !!!
        # Terminology: two arrays *overlap* if the lowermost memory addressed
        # touched by upper one is higher than the uppermost memory address
        # touched by the lower one.
        #
        # np.may_share_memory returns True iff there is overlap.
        # Overlap is a necessary but insufficient condition for *aliasing*.
        #
        # Aliasing is when two ndarrays refer a common memory location.
        if self.base is not other.base or self.size == 0 or other.size == 0:
            return False
        elif self is other or self.same_view_as(other):
            return True
        elif self.ndim < other.ndim:
            return other.shares_memory_with(self)

        assert self.ndim > 0
        if self.ndim == 1:
            # -- self is a vector view
            #    and other is either a scalar or vector view
            ae0, = self.elemstrides
            be0, = other.elemstrides
            amin = self.offset
            amax = amin + self.shape[0] * ae0
            bmin = other.offset
            bmax = bmin + other.shape[0] * be0
            if amin <= amax <= bmin <= bmax or bmin <= bmax <= amin <= amax:
                return False
            elif ae0 == be0 == 1:
                # -- strides are equal, and we've already checked for
                #    non-overlap. They do overlap, so they are aliased.
                return True
            # TODO: look for common divisor of ae0 and be0
            raise NotImplementedError('1d', (self.structure, other.structure))
        elif self.ndim == 2:
            # -- self is a matrix view
            #    and other is either a scalar, vector or matrix view
            a_contig, amin, amax = self.is_contiguous()
            b_contig, bmin, bmax = other.is_contiguous()

            if a_contig and b_contig:
                # -- both have a contiguous memory layout,
                #    from min up to but not including max
                return (not (amin <= amax <= bmin <= bmax)
                        and not (bmin <= bmax <= amin <= amax))
            elif a_contig:
                # -- only a contiguous
                raise NotImplementedError('2d self:contig, other:discontig',
                                          (self.structure, other.structure))
            else:
                raise NotImplementedError('2d',
                                          (self.structure, other.structure))
        raise NotImplementedError()


class Signal(SignalView):
    """Interpretable, vector-valued quantity within Nengo"""

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(self, value, name=None):
        self.value = np.asarray(value, dtype=np.float64)
        if name is not None:
            self._name = name
        if Signal.assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", shape=" + str(self.shape) + ")"
        except AttributeError:
            return ("Signal(id " + str(id(self)) + ", shape="
                    + str(self.shape) + ")")

    def __repr__(self):
        return str(self)

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def elemstrides(self):
        s = np.asarray(self.value.strides)
        return tuple(int(si / self.dtype.itemsize) for si in s)

    @property
    def offset(self):
        return 0

    @property
    def base(self):
        return self

