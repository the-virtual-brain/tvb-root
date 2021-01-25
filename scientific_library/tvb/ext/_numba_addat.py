import numba


@numba.jit(
    numba.void(numba.double[:, :, :], numba.long_[:], numba.double[:, :, :]),
    nopython=True,
    cache=True,
)
def add_at_313(state_reg, rmap, state):
    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    # this is costly, and dropped by the unsafe version
    for vi in range(nvert):
        reg_id = rmap[vi]
        assert 0 <= reg_id < nreg

    for vi in range(nvert):
        reg_id = rmap[vi]

        for svi in range(nsv):
            for mi in range(nmode):
                state_reg[reg_id, svi, mi] += state[vi, svi, mi]

@numba.jit(
    numba.void(numba.double[:, :, :], numba.long_[:], numba.double[:, :, :]),
    nopython=True,
    cache=True,
)
def add_at_313_unsafe(state_reg, rmap, state):
    """
    you *MUST* GUARANTEE the following or we SEGFAULT !

    assert 0 <= rmap[i] < nreg
    """
    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    for vi in range(nvert):
        reg_id = rmap[vi]

        for svi in range(nsv):
            for mi in range(nmode):
                state_reg[reg_id, svi, mi] += state[vi, svi, mi]

