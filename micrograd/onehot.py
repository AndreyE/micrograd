from collections import defaultdict
import torch
import numpy as np

from micrograd.engine import Value

_NEG = -1.0
VOID = 0.0
_POS = 1.0

def vals2tensor(values):
    if isinstance(values, torch.Tensor):
        return values
    if len(values):
        if isinstance(values[0], list):
            if len(values[0]):
                return torch.tensor([[v.data for v in vs] for vs in values])
        elif isinstance(values[0], Value):
            return torch.tensor([v.data for v in values])
    assert False


def arr2vals(array, name=''):
    return [[Value(array[i, j].item(), _name=f'{name}:{i}:{j}') for j in range(array.shape[1])] for i in range(array.shape[0])]


def batch2vals(batch):
    return tuple(map(arr2vals, batch))


def oh_encode(classes: torch.tensor):
    if isinstance(classes, list):
        classes = torch.tensor(classes)

    assert classes.dim() == 1
    DIM = classes.shape[0]

    void = torch.tensor([VOID] * len(classes))

    oh_enc = defaultdict(lambda: void)
    oh_dec = defaultdict(lambda: None)

    for i, cls in enumerate(classes):
        cls_vec = torch.tensor([_NEG] * classes.shape[0])
        cls_vec[i] = _POS
        oh_enc[cls.item()] = cls_vec
        oh_dec[tuple(cls_vec.numpy())] = cls.item()

    def oh_decoder(encoded):
        encoded = vals2tensor(encoded)
        encoded = np.round(encoded)
        # pick the closest vector (is round() OK for that?)
        if len(encoded.shape) > 1:
            return [oh_dec[tuple(row.numpy())] for row in encoded]
        elif len(encoded.shape) == 1:
            assert encoded.shape[-1] == DIM, f'unfit OH encoding {encoded.shape[-1]} != {DIM} (targets dimention)'
            return oh_dec[tuple(encoded.numpy())]
        assert False

    def oh_encoder(Y):
        if isinstance(Y, list):
            Y = torch.tensor(Y)
        return torch.vstack([oh_enc[cls.item()] for cls in Y])

    return oh_encoder, oh_decoder
