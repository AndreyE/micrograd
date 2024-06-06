import random
import numpy as np
from micrograd.engine import Value

class Module:
    def parameters(self):
        return []

class Neuron(Module):
    EPS = 1e-3

    def __init__(self, nin, act, bias=False, init=lambda: 0.0, _lid='', _nid='', **kwargs):
        self.id = f'L{_lid}:n{_nid}->{act}({nin})'
        self.w = [Value(init(), _name=f'L{_lid}:n{_nid}:w{w}', **kwargs) for w in range(nin)]
        self.b = Value(init(), _name=f'L{_lid}:n{_nid}:b', **kwargs) if bias else None

        self._train = False
        self._activate = self._pick_activation(act, debug=False)
        self._lid = _lid
        self._nid = _nid
        self._norm = 0.0

    def __call__(self, X):
        outs = []
        for xi in X:
            assert len(xi), xi
            outs.append(self._activate(xi))

        if self._train:
            data = np.array([o.data for o in outs])
            self._minmax = (data.min(), data.max())
            space = self._minmax[1] - self._minmax[0]
            print(f'debug: minmax = {self._minmax}, space = {space}')

            if self._norm > 1.0:
                for out in outs:
                    out /= space

        return outs

    def set_train(self, train: bool):
        if self._activate == self._line:
            self._train = train

    def _pick_activation(self, act='line', debug=False):
        act_f = None
        if act == 'line':
            act_f = self._line
        elif act == 'sbin':
            act_f = self._sbin
        elif act == 'bin':
            act_f = self._bin
        elif act == 'minmax':
            act_f = self._minmax
        elif act == 'snap':
            act_f = self._snap

        assert act_f is not None, f'Unsupported activation function {act}'

        if debug:
            def debug_act_f(x):
                act = act_f(x)

                backward = act._backward
                def _backward():
                    backward()
                    print(f'debug: {act._name}:{act._op}:[{act.data}, {act.grad}]')
                act._backward = _backward

                return act

            return debug_act_f

        return act_f

    def _line(self, x):
        assert len(self.w) == len(x), f'w[{len(self.w)}] != x[{len(x)}])'

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

        act = act / abs(act.data)
        act._name = 'sbin'
        return act

    def _bin(self, x):
        act = (self._sbin(x) + 1) / 2
        act._name = 'bin'
        act.data = np.round(act.data) # round to {0, 1}
        return act

    def _snap(self, x): # WARN: not working
        act = self._line(x)
        if abs(act.data) > 0.5:
            act = act / abs(act.data)
        else:
            act *= 0.0
        return act

    def _minmax(self, x):
        params = np.array([p.data for p in self.parameters()])
        minmax = params.max() - params.min()

        if minmax < Neuron.EPS:
            out = self._line(x)
        else:
            out = self._line(x) / minmax
            out._name = 'minmax'
        return out

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

    def __call__(self, X):
        outs = [n(X) for n in self.neurons]
        outs = list(zip(*outs))
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer L{self._lid} of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, layers, train):
        self.layers = layers
        self.set_train(train)

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def set_train(self, train: bool):
        self._train = train
        for n in self.neurons():
            n.set_train(train)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def neurons(self):
        return [n for layer in self.layers for n in layer.neurons]

    def learn_from(self, loss: Value, q: float = 1.0, logging=False, LR=None):
        # propagate grad
        loss.backward(logging=logging)
        # learn
        for p in self.parameters():
            p.learn(q=q, logging=logging, LR=LR)

    def make_learner(self, X, get_loss, LR=None):
        scores = self(X)
        current_loss = get_loss(scores)

        def learner(i=1, q=1.0, logging=False):
            nonlocal current_loss
            nonlocal scores
            nonlocal LR

            for k in range(i):
                print(f'{k} loss: {current_loss.data}')
                self.learn_from(current_loss, logging=logging, q=q, LR=LR)

                scores = self(X)
                current_loss = get_loss(scores)

            print(f'final loss: {current_loss.data}')
            return current_loss, scores

        return learner

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
