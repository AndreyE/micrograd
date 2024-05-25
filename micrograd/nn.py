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
        self.w = [Value(random.uniform(-1,1), **kwargs) for _ in range(nin)]
        self.b = Value(0, **kwargs)
        self.act = act
        self.frozen = False

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.act == 'relu':
            return act.relu()
        elif self.act == 'tanh':
            return act.tanh()

        assert self.act == 'linear'
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.act}-Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        self.frozen = False

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, **kwargs):
        sz = [(nin, None)] + nouts
        self.layers = [
            Layer(
                sz[i][0],
                sz[i+1][0], # layer dimension
                act=sz[i+1][1], # activation function
                **kwargs
            )
            for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def learn(self, q=0.5):
        for p in self.parameters():
            p.learn()
        self.zero_grad()

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
