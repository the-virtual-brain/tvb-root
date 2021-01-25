import _addat
import _numba_addat
import numpy as np


def init():
    state_reg = np.zeros((nreg, nsv, nmode))
    rmap = np.arange(nvert) % nreg
    # this will segfault unsafe versions:
    # rmap[2] = nreg + 3
    state = np.ones((nvert, nsv, nmode)) * 0.0001
    return state_reg, rmap, state


def tst_numpy():
    state_reg, rmap, state = init()
    for t in range(nticks):
        np.add.at(state_reg, rmap, state)
    return state_reg


def tst_cython_unsafe():
    state_reg, rmap, state = init()
    for t in range(nticks):
        _addat.add_at_313_unsafe(state_reg, rmap, state)
    return state_reg


def tst_cython_unsafe_ccontig():
    state_reg, rmap, state = init()
    for t in range(nticks):
        _addat.add_at_313_c_contig_unsafe(state_reg, rmap, state)
    return state_reg


def tst_numba():
    state_reg, rmap, state = init()
    for t in range(nticks):
        _numba_addat.add_at_313(state_reg, rmap, state)
    return state_reg


def tst_numba_unsafe():
    state_reg, rmap, state = init()
    for t in range(nticks):
        _numba_addat.add_at_313_unsafe(state_reg, rmap, state)
    return state_reg


def _runall():
    # numpy_base = tst_numpy()
    c_unsafe = tst_cython_unsafe()
    c_unsafe_ccontig = tst_cython_unsafe_ccontig()
    numba_res = tst_numba()
    numba_unsafe_res = tst_numba_unsafe()

    numpy_base = numba_res

    assert np.abs(c_unsafe - numpy_base).max() < 1e-12
    assert np.abs(c_unsafe_ccontig - numpy_base).max() < 1e-12
    assert np.abs(numba_res - numpy_base).max() < 1e-12
    assert np.abs(numba_unsafe_res - numpy_base).max() < 1e-12


def test_accelerator_module():
    global nvert, nreg, nsv, nmode, nticks
    # small for debug
    nreg = 8
    nsv = 3
    nvert = 27
    nmode = 2
    nticks = 5

    _runall()


def perf_test():
    global nvert, nreg, nsv, nmode, nticks
    # large for profile
    nvert = 21920
    nreg = 85
    nsv = 3
    nmode = 2
    nticks = 5000

    _runall()


if __name__ == '__main__':
    # test_accelerator_module()
    perf_test()
