import numpy as np
import logging


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=[], _lr=1.0, _op='', _name='auto'):
        self.data = np.float64(data)
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = _children
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._name = _name
        self._lr = _lr
        self._pgrad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other,)
        out = Value(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __sub__(self, other): # self - other
        return self + -other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), _op=f'^{other}')

        def _backward():
            if self.data != 0.0:
                self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def child(self, idx):
        return self._prev[idx]

    def zero(self):
        out = self - self
        out._op = 'zero'
        return out

    def relu(self):
        y = 0 if self.data < 0 else self.data
        out = Value(y, (self,), _op='ReLU')

        def _backward():
            self.grad += (y > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = Value(y, (self,), _op='tanh')

        def _backward():
            self.grad += (1 - y**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        y = np.exp(self.data)
        out = Value(y, (self, ), _op='exp')

        def _backward():
            self.grad = y * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        y = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(y, (self, ), _op='Sigmoid')

        def _backward():
            self.grad += y * (1 - y) * out.grad
        out._backward = _backward

        return out

    def abs(self):
        return (self**2)**0.5

    def backward(self, logging=False):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            v.zero_grad()

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
            if logging:
                print(f'backward:{v._name}:[grad <- {v.grad}]')

    def zero_grad(self):
        self._pgrad = self.grad
        self.grad = 0.0

    def learn(self, q=1.0, logging=False, LR=None):
        assert q <= 1

        # if previous gradient has different sign then reduce the step size by q
        lr = LR
        if lr is None:
            if self._pgrad * self.grad < 0:
                self._lr *= q
            else:
                self._lr *= q  ** -(1/2)
            lr = self._lr

        data = self.data - lr * self.grad

        if logging:
            print(f'learn:{self._name}[data[{data} <- {self.data}+{-(lr * self.grad)}]')
        self.data = data

    def __neg__(self): # -self
        out = self * Value(-1, _name='-1')
        out._op = '*(-1)'
        out._name = f'neg-{self._name}'
        return out

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        out = self * other**-1
        out._op = '*'
        out._name = 'div'
        return out

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value({self._name} : [{self.data}, {self.grad}, {self._lr}])"
