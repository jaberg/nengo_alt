
from .signaldict import SignalDict
from .graph_algos import toposort
from .depgraph import operator_depencency_graph


class Engine(object):

    def _make_step(self, node):
        return node.make_step(self.signals)

    def _init_signals(self, node):
        node.init_signals(self.signals)

    def __init__(self, operators, signals=None):
        """

        operators: list of Operator instances

        signals: SignalDict instance

        """
        self.signals = SignalDict() if signals is None else signals
        map(self._init_signals, operators)
        self.dg = operator_depencency_graph(operators)
        self._step_order = [node for node in toposort(self.dg)
                            if hasattr(node, 'make_step')]
        self._steps = map(self._make_step, self._step_order)
        self.n_steps = 0

    def step(self, N=1):
        """Simulate for the given number of `dt` steps."""
        for ii in range(N):
            for step_fn in self._steps:
                step_fn()
            self.n_steps += 1

        # post-condition:
        # self.signals[signal] reveals current value of signal in model
        # as read-only numpy ndarray (see SigDict for details)

