import torch

def puzzle2d(x, shape: tuple[int, int], slide=None):
    assert len(shape) == 2, f'shape = {shape}'
    assert isinstance(shape[0], int) and shape[0] > 0, f'shape = {shape}'
    assert isinstance(shape[1], int) and shape[1] > 0, f'shape = {shape}'

    if slide is None:
        slide = (
            shape[0] // 2 if shape[0] > 1 else 1,
            shape[1] // 2 if shape[1] > 1 else 1
        )
    else:
        assert slide[0] <= shape[0], f'slide = {slide}'
        assert slide[1] <= shape[1], f'slide = {slide}'

    fragments = []
    for i in range(0, x.shape[0], slide[0]):
        for j in range(0, x.shape[1], slide[1]):
            fragment = x[i:i+shape[0], j:j+shape[1]]
            if fragment.shape == shape:
                fragments.append(fragment)
    return torch.stack(fragments)
