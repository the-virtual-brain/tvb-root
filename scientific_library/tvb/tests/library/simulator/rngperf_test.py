import numpy as np
import pytest

BitGens = [
    np.random.PCG64,
    np.random.Philox,
    np.random.MT19937,
    np.random.SFC64,
]

@pytest.mark.parametrize('BitGen', BitGens)
def test_raw(benchmark, BitGen):
    bg = BitGen(42)
    benchmark(lambda: bg.random_raw(size=1024, output=False))

@pytest.mark.parametrize('BitGen', BitGens)
def test_uniform(benchmark, BitGen):
    bg = BitGen(42)
    rng = np.random.Generator(bg)
    out = np.empty((1024,), 'f')
    benchmark(lambda: rng.random(out.shape, np.float32, out))

@pytest.mark.parametrize('BitGen', BitGens)
def test_normal(benchmark, BitGen):
    bg = BitGen(42)
    rng = np.random.Generator(bg)
    benchmark(lambda: rng.normal(0.0, 1.0, 1024))

