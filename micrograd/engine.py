import numpy as np

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', _name='auto', _lr=1.0):
        self.data = data
        self.grad = 0.0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self._name = _name
        self._lr = _lr
        self._pgrad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

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
        out = Value(self.data**other, (self,), _op=f'**{other}')

        def _backward():
            if self.data != 0.0:
                self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

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

    def squeeze(self):
        out = ((self - self._min) / self._space) * 2 - 1
        out._op = 'squeeze'
        return out

    def xspace(self):
        out = self / abs(self.data)
        out._op = 'xspace'
        return out

    def pxspace(self):
        out = (self.xspace() + 1) / 2
        out._op = '+xspace'
        return out

    def backward(self):

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

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self._pgrad = self.grad
        self.grad = 0.0

    def learn(self, q=0.5):
        assert q <= 1

        if self.grad == 0.0:
            return

        # keep the stride stable
        # if self._pgrad > 0.0:
        #    self._lr = abs(self._lr * self._pgrad / self.grad)

        # if previous gradient has different sign then reduce the step size by q
        if self._pgrad * self.grad < 0:
            self._lr *= q
        else:
            self._lr *= q  ** -(1/2)

        self.data -= self._lr * self.grad

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value({self._name} : [{self.data}, {self.grad}])"
