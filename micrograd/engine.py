import numpy as np
import logging


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=[], LR=1.0, _op='', _name='auto', _frozen=False):
        self.data = np.float64(data)
        self._grad = []

        self._frozen = _frozen
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
        self._name = _name
        self._lr = LR
        self._pgrad = 0.0

    def grad(self):
        return sum(self._grad)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other,)
        out = Value(self.data + other.data, (self, other), _op='+')

        def _backward():
            self._grad += [1 * out.grad()]
            other._grad += [1 * out.grad()]
        out._backward = _backward

        return out

    def __sub__(self, other): # self - other
        return self + -other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op='*')

        def _backward():
            self._grad += [other.data * out.grad()]
            other._grad += [self.data * out.grad()]
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), _op=f'^{other}')

        def _backward():
            if self.data != 0.0:
                self._grad += [(other * self.data**(other-1)) * out.grad()]
        out._backward = _backward

        return out

    def freeze(self):
        self._frozen = True

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
            self._grad += [(y > 0) * out.grad()]
        out._backward = _backward

        return out

    def tanh(self):
        y = np.tanh(self.data)
        out = Value(y, (self,), _op='tanh')

        def _backward():
            self._grad += [(1 - y**2) * out.grad()]
        out._backward = _backward

        return out

    def exp(self):
        y = np.exp(self.data)
        out = Value(y, (self, ), _op='exp')

        def _backward():
            self._grad = y * out.grad()
        out._backward = _backward

        return out

    def sigmoid(self):
        y = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(y, (self, ), _op='Sigmoid')

        def _backward():
            self._grad += [y * (1 - y) * out.grad()]
        out._backward = _backward

        return out

    def abs(self):
        if self.data >= 0:
            return self
        return self * -1.0

    def sbin(self):
        if abs(self.data) == 0.0:
            return self
        out = self / abs(self.data)
        out.data = np.round(out.data)
        out._name = 'sbin'
        return out

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
        self._grad = [1]
        for v in reversed(topo):
            v._backward()
            if logging:
                print(f'backward:{v._name}:[grad <- {v.grad}]')

    def zero_grad(self):
        self._pgrad = self.grad()
        self._grad = []

    def learn(self, q=1.0, logging=False, LR=None):
        assert q <= 1

        if self._frozen:
            return False

        # if previous gradient has different sign then reduce the step size by q
        grad = self.grad()
        lr = LR
        if lr is None:
            if self._pgrad * grad < 0:
                self._lr *= q
            else:
                self._lr *= q  ** -(1/2)
            lr = self._lr

        change = lr * grad
        data = self.data - change

        if logging:
            print(f'learn:{self._name}[data[{data} <- {self.data}+{-(lr * grad)}]')
        self.data = data

        return change != 0.0

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
        return f"Value({self._name} : [{self.data}, {self._grad}, {self._lr}])"
