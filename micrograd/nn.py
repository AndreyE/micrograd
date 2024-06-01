import random
import numpy as np
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, act, init=lambda: random.uniform(-1,1), **kwargs):
        self.w = [Value(init(), _name='weight', **kwargs) for _ in range(nin)]
        self.b = Value(init(), _name='bias', **kwargs)
        self.act = act
        self._history = []

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        if self.act == 'relu':
            return act.relu()
        elif self.act == 'tanh':
            return act.tanh()
        elif self.act == 'sigmoid':
            return act.sigmoid()
        elif self.act == 'squeeze':
            return act.squeeze()
        elif self.act == 'xspace':
            return act.xspace()
        elif self.act == '+xspace':
            return act.pxspace()
        elif self.act == 'minmax':
            key = lambda p: p.data
            minmax = max(self.parameters(), key=key) - min(self.parameters(), key=key)
            return act / minmax

        assert self.act == 'linear'
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.act}-Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, lr, **kwargs):
        sz = [(nin, None)] + nouts
        self.layers = [
            Layer(
                sz[i][0],
                sz[i+1][0],
                act=sz[i+1][1],
                _lr=lr,
                **kwargs
            )
            for i in range(len(nouts))
        ]
        # self.norm()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def learn_from(self, loss: Value, q: float = 0.5, norm=True):
        # propagate grad
        self.zero_grad()
        loss.backward()
        # learn
        for p in self.parameters():
            p.learn(q)

        if norm:
            self.norm()

    def norm(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                params = np.array([p.data for p in neuron.parameters()])
                norm = np.std(params)
                if norm > 0.0:
                    for p in neuron.parameters():
                        p.data /= norm
                        # p._lr /= norm

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
