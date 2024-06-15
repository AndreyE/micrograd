import random
import numpy as np
from micrograd.engine import Value

class Module:
    def parameters(self):
        return []

class Neuron(Module):
    EPS = 1e-3

    def __init__(self, act, bias=False, init=lambda: 0.0, _lid='', _nid='', **kwargs):
        self.act = act
        self._nin = 0
        self._train = False
        self._init = init
        self._activate = self._pick_activation(act, debug=False)
        self._lid = _lid
        self._nid = _nid
        self._kwargs = kwargs

        self.w = []
        self.b = Value(init(), _name=f'L{_lid}:n{_nid}:b', **kwargs) if bias else None

    def __call__(self, X):
        acts = []
        for xi in X:
            assert len(xi), xi
            acts.append(self._activate(xi))

        if self._activate == self._line:
            data = np.array([act.data for act in acts])
            self._minmax = (data.min(), data.max())
            space = self._minmax[1] - self._minmax[0]
            # print(f'debug: minmax = {self._minmax}, space = {space}')

            if space != 0.0:
                return [act / space for act in acts]

        return acts

    def _accomodate(self, X):
        self.w.extend([
            Value(
                    self._init(),
                    _name=f'L{self._lid}:n{self._nid}:w{w}',
                    **self._kwargs
                )
                for w in range(len(self.w), len(X))
        ])

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

        def prod_act_f(X):
            self._accomodate(X)
            return act_f(X)

        return prod_act_f

    def _line(self, x):
        assert len(self.w) == len(x), f'w[{len(self.w)}] != x[{len(x)}])'

        act = self.w[0] * x[0]
        for i in range(1, len(self.w)):
            act += self.w[i] * x[i]

        if self.b is not None:
            act += self.b

        act._name = 'line'
        act._op = 'line'
        return act

    def _sbin(self, x):
        act = self._line(x)
        if abs(act.data) < Neuron.EPS: # make it `line` if too close to zero
            return act
        return act.sbin()

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

    def freeze(self):
        for p in self.parameters():
            p.freeze()

    def learn(self, q: float = 1.0, logging=False, LR=None):
        return any([p.learn(q=q, logging=logging, LR=LR) for p in self.parameters()])

    def __repr__(self):
        return f'L{self._lid}:n{self._nid}->{self.act}({self._nin})'

class Layer(Module):

    def __init__(self, act, _nout=1, _lid=None, LR=1.0, **kwargs):
        self.neurons = [Neuron(act, _nid=n, _lid=_lid, LR=LR, **kwargs) for n in range(_nout)]
        self._lr = LR
        self._lid = _lid
        self._act = act
        self._kwargs = kwargs
        self.expandable=True

    def __call__(self, X):
        outs = [n(X) for n in self.neurons]
        outs = list(zip(*outs))
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def learn(self, q: float = 1.0, logging=False, LR=None):
        return any([n.learn(q=q, logging=logging, LR=LR) for n in self.neurons])

    def freeze(self):
        for n in self.neurons:
            n.freeze()

    def expand(self):
        if self.expandable:
            self.neurons.append(
                Neuron(self._act, _nid=len(self.neurons), _lid=self._lid, **self._kwargs)
            )

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
        learnt_smth = False
        # propagate grad
        loss.backward(logging=logging)
        # learn
        for l in self.layers[:-1]:
            if l.learn(q=q, logging=logging, LR=LR):
                # l.freeze()
                learnt_smth = True
                l.expand()
        return self.layers[-1].learn(q=q, logging=logging, LR=LR) or learnt_smth

    def make_learner(self, X, get_loss, ESAT=0.0, LR=None):
        scores = self(X)
        current_loss = get_loss(scores)

        def learner(i=1, q=1.0, logging=False):
            nonlocal current_loss
            nonlocal scores
            nonlocal LR
            nonlocal ESAT

            for k in range(i):
                print(f'{k} loss: {current_loss.data}')
                learning_progress = self.learn_from(current_loss, logging=logging, q=q, LR=LR)

                scores = self(X)
                current_loss = get_loss(scores)
                if current_loss.data <= ESAT:
                    print(f'EARLY STOP BY ESAT={ESAT}!')
                    break

                if not learning_progress:
                    print(f'SUDDEN STOP BECAUSE OF NO LEARNING PROGRESS!')
                    break

            print(f'final loss: {current_loss.data}')
            return current_loss, scores

        return learner, scores, current_loss

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
