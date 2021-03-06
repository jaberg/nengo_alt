import inspect
import logging

import pytest

import nengo
from nengo.utils.decorators import decorator, memoize

logger = logging.getLogger(__name__)
state = None  # Used to make sure decorators are running


def _test_decorated(obj):
    global state

    # Make sure decorator works
    state = 'not run'
    obj(0, 0)
    assert state == 'run'
    state = 'not run'

    # Make sure decorated function looks like non-decorated
    assert obj.__name__ == 'f'
    assert obj.__doc__ == "Return 1."


def test_function():
    """Test that @decorator works on plain functions."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert instance is None
        return wrapped(*args, **kwargs)

    @test_decorator
    def f(a, b):
        """Return 1."""
        return 1

    _test_decorated(f)

    # Make sure introspection works
    assert inspect.getargspec(f).args == ['a', 'b']
    assert inspect.getsource(f) == ('    @test_decorator\n'
                                    '    def f(a, b):\n'
                                    '        """Return 1."""\n'
                                    '        return 1\n')


def test_boundfunction():
    """Test that @decorator works on bound functions."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert instance is not None
        assert instance.__class__.__name__ == 'Test'
        return wrapped(*args, **kwargs)

    class Test(object):
        @test_decorator
        def f(self, a, b):
            """Return 1."""
            return 1

    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getargspec(inst.f).args == ['self', 'a', 'b']
    assert inspect.getsource(inst.f) == ('        @test_decorator\n'
                                         '        def f(self, a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')


def test_staticmethod():
    """Test that @decorator works on staticmethods."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert instance is None
        return wrapped(*args, **kwargs)

    # --- Decorator before staticmethod
    class Test(object):
        @test_decorator
        @staticmethod
        def f(a, b):
            """Return 1."""
            return 1

    _test_decorated(Test.f)
    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getargspec(inst.f).args == ['a', 'b']
    assert inspect.getsource(inst.f) == ('        @test_decorator\n'
                                         '        @staticmethod\n'
                                         '        def f(a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')

    # --- Decorator after staticmethod
    class Test(object):
        @staticmethod
        @test_decorator
        def f(a, b):
            """Return 1."""
            return 1

    _test_decorated(Test.f)
    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getargspec(inst.f).args == ['a', 'b']
    assert inspect.getsource(inst.f) == ('        @staticmethod\n'
                                         '        @test_decorator\n'
                                         '        def f(a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')


def test_classmethod():
    """Test that @decorator works on classmethods."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert (instance is None and args[0].__name__ == 'Test'
                or instance.__name__ == 'Test')
        return wrapped(*args, **kwargs)

    # --- Decorator before classmethod
    class Test(object):
        @test_decorator
        @classmethod
        def f(cls, a, b):
            """Return 1."""
            return 1

    _test_decorated(Test.f)
    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getargspec(inst.f).args == ['cls', 'a', 'b']
    assert inspect.getsource(inst.f) == ('        @test_decorator\n'
                                         '        @classmethod\n'
                                         '        def f(cls, a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')

    # --- Decorator after staticmethod
    class Test(object):
        @classmethod
        @test_decorator
        def f(cls, a, b):
            """Return 1."""
            return 1

    _test_decorated(Test.f)
    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getargspec(inst.f).args == ['cls', 'a', 'b']
    assert inspect.getsource(inst.f) == ('        @classmethod\n'
                                         '        @test_decorator\n'
                                         '        def f(cls, a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')


def test_class():
    """Test that @decorator works on classes."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert instance is None
        inst = wrapped(*args, **kwargs)
        inst.ran = True
        return inst

    @test_decorator
    class f(object):
        """Return 1."""
        def __init__(self, a, b):
            self.a = a
            self.b = b

    _test_decorated(f)
    inst = f('a', 'b')
    assert inst.a == 'a' and inst.b == 'b'
    assert inst.ran
    assert inst.__class__ == f.__wrapped__
    assert type(inst) == f.__wrapped__

    # Make sure introspection works
    # Note: for classes, the decorator isn't part of the source. Weird!
    assert inspect.getsource(f) == ('    class f(object):\n'
                                    '        """Return 1."""\n'
                                    '        def __init__(self, a, b):\n'
                                    '            self.a = a\n'
                                    '            self.b = b\n')


def test_memoize():
    """Test that the memoize decorator works in several contexts."""
    cache_info = {'misses': 0}  # must be a dict so that closure happens

    @memoize
    def f():
        cache_info['misses'] += 1
        return 'f'

    @memoize
    def f_args(a, b=5):
        cache_info['misses'] += 1
        return a + b

    class Test(object):
        @memoize
        def inst_f(self):
            cache_info['misses'] += 1
            return 'inst_f'

        @memoize
        @classmethod
        def cls_f(cls):
            cache_info['misses'] += 1
            return 'cls_f'

        @memoize
        @staticmethod
        def static_f():
            cache_info['misses'] += 1
            return 'static_f'

        @property
        @memoize
        def prop(self):
            cache_info['misses'] += 1
            return 'prop'

    def check_all(inst):
        assert f() == f.__name__
        assert inst.inst_f() == inst.inst_f.__name__
        assert Test.cls_f() == Test.cls_f.__name__
        assert Test.static_f() == Test.static_f.__name__
        assert inst.prop == 'prop'
        assert f_args(1) == 6
        assert f_args(a=10) == 15
        assert f_args(1, 2) == 3
        assert f_args(1, b=100) == 101

    inst = Test()

    # First run should be all misses
    check_all(inst)
    assert cache_info['misses'] == 9

    # Second run should be all hits
    check_all(inst)
    assert cache_info['misses'] == 9


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
