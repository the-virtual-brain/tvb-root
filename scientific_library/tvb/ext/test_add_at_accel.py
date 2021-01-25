import _addat
import numpy as np

nreg = 8
nsv = 3
nvert = 27
nmode = 2


def init():
    state_reg = np.zeros((nreg, nsv, nmode))
    rmap = np.arange(nvert) % nreg
    state = np.ones((nvert, nsv, nmode))
    return state_reg, rmap, state


def tst1():
    state_reg, rmap, state = init()
    np.add.at(state_reg, rmap, state)
    return state_reg


def tst2():
    state_reg, rmap, state = init()
    _addat.add_at_313_unsafe(state_reg, rmap, state)
    return state_reg


def tst3():
    state_reg, rmap, state = init()
    _addat.add_at_313_c_contig_unsafe(state_reg, rmap, state)
    return state_reg


def test_accelerator_module():
    numpy_base = tst1()
    c_unsafe = tst2()
    c_unsafe_ccontig = tst3()

    assert np.abs(c_unsafe - numpy_base).max() < 1e-12
    assert np.abs(c_unsafe_ccontig - numpy_base).max() < 1e-12

if __name__ == '__main__':
    test_accelerator_module()
