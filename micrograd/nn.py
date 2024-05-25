import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def freeze(self):
        for layer in self.layers:
            if not layer.freeze():
                return False
        return True

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, act='linear'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
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

    def freeze(self):
        if not self.frozen:
            for v in self.parameters():
                if v.learning_rate * abs(v.grad) < 1e-4:
                    v.learning_rate = 0.0
            self.frozen = all([v.learning_rate == 0.0 for v in self.parameters()])
        return self.frozen

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

    def freeze(self):
        if not self.frozen:
            for n in self.neurons:
                n.freeze()
            self.frozen = all([n.frozen for n in self.neurons])
        return self.frozen

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [(nin, None)] + nouts
        self.layers = [
            Layer(
                sz[i][0],
                sz[i+1][0], # layer dimension
                act=sz[i+1][1] # activation function
            )
            for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
