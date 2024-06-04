from micrograd.engine import Value
from micrograd.onehot import arr2vals


def calc_accuracy(scores, targets, oh_decoder):
    matches = 0
    for score, target in zip(scores, targets):
        matches += oh_decoder(score) == oh_decoder(target)
    return matches / len(scores)


def one_sq_loss(score, targets):
    assert len(score) == len(targets)
    out = sum([(s - t).abs() for s, t in zip(score, targets)], start=Value(0.0, _name='init loss'))
    out._name = f'one loss'
    return out


def calc_sq_loss(scores, targets):
    assert len(scores) == len(targets), f'{len(scores)} != {len(targets)}'
    out = sum([one_sq_loss(xs, ts) for xs, ts in zip(scores, targets)], start=Value(0.0, _name='init loss'))
    out._name = 'loss'
    return out


def evaluate(model, X, Y_oh, oh_decoder):
    input = arr2vals(X, 'input')
    scores = model(input)
    targets = arr2vals(Y_oh, 'expected')
    return calc_sq_loss(scores, targets), calc_accuracy(scores, targets, oh_decoder), scores, targets
