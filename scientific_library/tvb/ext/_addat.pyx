# cython: language_level=3

cimport cython


@cython.boundscheck(False)
cdef void _add_at_313(
        double[:, :, :] state_reg,
        const long[:] rmap,
        const double[:, :, :] state,
        size_t nreg,
        size_t nsv,
        size_t nvert,
        size_t nmode,
) nogil:
    cdef long reg_id

    for vi in range(nvert):
        reg_id = rmap[vi]

        for svi in range(nsv):
            for mi in range(nmode):
                state_reg[reg_id, svi, mi] += state[vi, svi, mi]


# arr[:, ::1] declares it as C contiguous in all dimensions

@cython.boundscheck(False)
cdef void _add_at_313_c_contig(
        double[:, :, ::1] state_reg,
        const long[:] rmap,
        const double[:, :,::1] state,
        size_t nreg,
        size_t nsv,
        size_t nvert,
        size_t nmode,
) nogil:
    cdef long reg_id

    for vi in range(nvert):
        reg_id = rmap[vi]

        for svi in range(nsv):
            for mi in range(nmode):
                state_reg[reg_id, svi, mi] += state[vi, svi, mi]


def add_at_313(
        double[:, :, :] state_reg,
        const long[:] rmap,
        const double[:, :, :] state
    ):
    cdef size_t nreg, nsv, nvert, nmode

    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    cdef long reg_id

    for vi in range(nvert):
        reg_id = rmap[vi]
        assert 0 <= reg_id < nreg

    _add_at_313(state_reg, rmap, state, nreg, nsv, nvert, nmode)


def add_at_313_unsafe(
        double[:, :, :] state_reg,
        const long[:] rmap,
        const double[:, :, :] state
    ):
    """ you must guarantee that 0 <= rmap[i] < state_reg.shape[0] or we segfault """
    cdef size_t nreg, nsv, nvert, nmode

    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    cdef long reg_id

    _add_at_313(state_reg, rmap, state, nreg, nsv, nvert, nmode)



def add_at_313_c_contig_unsafe(
        double[:, :, ::1] state_reg,
        const long[:] rmap,
        const double[:, :, ::1] state
    ):
    """ you must guarantee that 0 <= rmap[i] < state_reg.shape[0] or we segfault """
    cdef size_t nreg, nsv, nvert, nmode

    nreg = state_reg.shape[0]
    nsv = state_reg.shape[1]
    nmode = state_reg.shape[2]
    nvert = state.shape[0]

    assert rmap.shape[0] == nvert
    assert state.shape[1] == nsv
    assert state.shape[2] == nmode

    cdef long reg_id

    _add_at_313_c_contig(state_reg, rmap, state, nreg, nsv, nvert, nmode)

