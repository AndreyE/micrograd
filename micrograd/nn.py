import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, act, **kwargs):
        self.w = [Value(random.uniform(-1,1), _name='weight', **kwargs) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1), _name='bias', **kwargs)
        self.act = act

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

    def __init__(self, nin, nouts, lr):
        sz = [(nin, None)] + nouts
        self.layers = [
            Layer(
                sz[i][0],
                sz[i+1][0],
                act=sz[i+1][1],
                _lr=lr
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def learn_from(self, loss: Value, q: float):
        # propagate grad
        self.zero_grad()
        loss.backward()
        # learn
        for p in self.parameters():
            p.learn(q)

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
