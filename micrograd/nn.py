import random
import numpy as np
from micrograd.engine import Value

class Module:
    def parameters(self):
        return []

class Neuron(Module):
    EPS = 1e-3

    def __init__(self, nin, act, bias=True, init=lambda: random.uniform(-1,1), _lid='', _nid='', **kwargs):
        self.id = f'L{_lid}:n{_nid}->{act}({nin})'
        self.w = [Value(init(), _name=f'L{_lid}:n{_nid}:w{w}', **kwargs) for w in range(nin)]
        self.b = Value(init(), _name=f'L{_lid}:n{_nid}:b', **kwargs) if bias else None

        self._activate = self._pick_activation(act)
        self._lid = _lid
        self._nid = _nid
        self._history = [] # TODO

    def __call__(self, x):
        return self._activate(x)

    def _pick_activation(self, act='line'):
        if act == 'line':
            return self._line
        elif act == 'sbin':
            return self._sbin
        elif act == 'bin':
            return self._bin
        elif act == 'minmax':
            return self._minmax

        assert False, f'Unsupported activation function {act}'

    def _line(self, x):
        assert len(self.w) == len(x), f'L{self._lid}:n{self._nid} <- {len(self.w)} != {len(x)}'

        act = self.w[0] * x[0]
        for i in range(1, len(self.w)):
            act += self.w[i] * x[i]

        if self.b is not None:
            act += self.b

        act._op = 'line'
        return act

    def _sbin(self, x):
        act = self._line(x)
        if abs(act.data) < Neuron.EPS: # if too close to zero
            return act

        act = act / act.abs()
        act._op = 'sbin'
        act.data = np.round(act.data) # round to {-1, 1}
        return act

    def _bin(self, x):
        act = (self._sbin(x) + 1) / 2
        act._op = 'bin'
        act.data = np.round(act.data) # round to {0, 1}
        return act

    def _minmax(self, x):
        act = self._line(x)
        if abs(act.data) < Neuron.EPS: # if too close to zero
            return act

        key = lambda p: p.data
        minmax = max(self.parameters(), key=key) - min(self.parameters(), key=key)
        return act / minmax

    def parameters(self):
        if self.b is None:
            return self.w
        else:
            return self.w + [self.b]

    def __repr__(self):
        return self.id

class Layer(Module):

    def __init__(self, shape, act, _lid=None, **kwargs):
        nin, nout = shape
        self.neurons = [Neuron(nin, act, _nid=n, _lid=_lid, **kwargs) for n in range(nout)]
        self._lid = _lid

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer L{self._lid} of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def learn_from(self, loss: Value, q: float = 1.0, logging=False, norm=False):
        # propagate grad
        loss.backward(logging=logging)
        # learn
        for p in self.parameters():
            p.learn(q=q, logging=logging)
        # normalize output
        if norm:
            self.norm()

    def norm(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                params = np.array([p.data for p in neuron.parameters()])
                norm = params.max() - params.min()
                if norm > 0.0:
                    for p in neuron.parameters():
                        p.data /= norm
                        # p._lr /= norm

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
