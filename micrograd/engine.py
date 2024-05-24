import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        # assert -1 <= data <= 1
        self.data = data
        self.grad = 0.0
        self.pgrad = 0.0
        self.min = min(-1, -abs(data))
        self.space = self.min * -2
        # internal variables used for autograd graph construction
        self._backward = lambda: 1
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        self.lr = 1.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def squeeze(self):
        out = ((self - self.min) / self.space) * 2 - 1
        out._op = 'squeeze'
        return out

    def topo(self):
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

        return topo

    def backward(self):
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(self.topo()):
            v._backward()

    def zero_grad(self):
        self.pgrad = self.grad
        self.grad = 0.0

    def learn(self, q = 0.5):
        assert q <= 1

        if self.grad == 0.0:
            return

        # keep the step size stable
        if abs(self.pgrad) > 0.0:
            self.lr = abs(self.lr * self.pgrad / self.grad)

        # if previous gradient has different sign then reduce the step size by q
        if self.pgrad * self.grad < 0:
            self.lr *= q
        else:
            self.lr *= q ** -(1/2)

        self.data -= self.lr * self.grad
        self.min = min(self.data, self.min)
        self.space = max(self.data - self.min, self.space)

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
        return f"Value(data={self.data}, min={self.min}, space={self.space}, pgrad={self.pgrad}, grad={self.grad}, lr={self.lr})"
