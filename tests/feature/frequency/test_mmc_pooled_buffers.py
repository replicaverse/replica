"""The per-cell priority buffers select the same candidates wherever they rest between merges."""

import numpy as np
import pytest
import torch

from replica.feature.frequency import pooled as P

#: Strata per cell in the test, independent of any study's stratification.
N_STRATA = 3

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _candidates(generator, n_cols, n, device):
    q = torch.rand(n_cols, n, dtype=torch.float64, generator=generator)
    q[torch.rand(n_cols, n, generator=generator) < 0.1] = -torch.inf
    p0 = torch.rand(n_cols, n, generator=generator).float()
    sample = torch.randint(0, 50, (1, n), generator=generator).expand(n_cols, n)
    pos = torch.randint(0, 40, (1, n), generator=generator).expand(n_cols, n)
    mult = torch.randint(1, 4, (1, n), generator=generator).expand(n_cols, n)
    return [t.contiguous().to(device) for t in (q, p0, sample, pos, mult)]


@pytest.mark.parametrize("device", DEVICES)
def test_the_buffers_rest_anywhere_without_changing_the_selection(device):
    generator = torch.Generator().manual_seed(11)
    capacity = torch.randint(0, 6, (7, N_STRATA, 2), generator=generator).numpy()
    resident = P._CellBuffers(capacity)
    hosted = P._CellBuffers(capacity, store_device="cpu")
    for _ in range(12):
        key = (
            int(torch.randint(0, N_STRATA, (1,), generator=generator)),
            int(torch.randint(0, 2, (1,), generator=generator)),
        )
        n = int(torch.randint(1, 9, (1,), generator=generator))
        chunk = _candidates(generator, 7, n, device)
        resident.offer(key, *chunk)
        hosted.offer(key, *chunk)
    for key, held in hosted.store.items():
        assert all(t.device.type == "cpu" for t in held.values())
        assert all(t.device.type == device for t in resident.store[key].values())
    left, right = resident.flatten(), hosted.flatten()
    assert left.keys() == right.keys()
    for name in left:
        assert np.array_equal(left[name], right[name]), name
    assert sum(int(c) for c in left["buf_count"]) > 0
