# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

# we need a global otherwise cython occasionally forgets to put a module init! A docstring will do it
"""
Accelerated history
"""
import numpy as np
cimport numpy as np
cimport cython

# Consider relaxing c alignment requirements.
# At this level we avoid allocating memory. Send buffers from python.
# The reason for the funny shapes like 1, ncvar, 1 is compatibility with the python version

def _get_state_unoptimized(
        np.ndarray[double, ndim=4] history not None, # t sv n m
        np.ndarray[int, ndim=3] time_idx not None, # n, 1, n
        np.ndarray[int, ndim=3] cvar not None, # 1, ncvar, 1
        np.ndarray[int, ndim=3] node_ids not None, #1, 1, n
        np.ndarray[double, ndim=4] delayed_states not None, #n ncvar n m
        ):
    """
    Uses the same code as the python version. Here to sanity check data marshalling and cython build.
    """
    delayed_states[...] = history[time_idx, cvar, node_ids, :]  #n ncvar n m

@cython.boundscheck(False)
@cython.wraparound (False)
cdef _common_bounds_check(
        np.ndarray[double, ndim=4] history,
        np.ndarray[int, ndim=3] time_idx,
        np.ndarray[int, ndim=3] cvar,
        np.ndarray[double, ndim=4] delayed_states
        ):
    cdef int horizon, nvars, nodes, ncvar, modes
    cdef int ns, cv, nd
    cdef int delay, cvar_idx
    horizon = history.shape[0]
    nvars = history.shape[1]
    nodes = history.shape[2]
    modes = history.shape[3]
    ncvar = cvar.shape[1]
    # bounds check time_idx[]
    if time_idx.shape[0] != nodes or time_idx.shape[1] != 1 or time_idx.shape[2] != nodes:
        raise IndexError()
    # bounds check delayed_states[]
    if (delayed_states.shape[0] != nodes or delayed_states.shape[1] != ncvar or
        delayed_states.shape[2] != nodes or delayed_states.shape[3] != modes):
        raise IndexError()
    # bounds check history[delay, cvar_idx] other dimensions safe by definition
    for cv in range(ncvar):
        cvar_idx = cvar[0, cv, 0]   # safe per ncvar definition
        if not 0 <= cvar_idx < nvars:
            raise IndexError()
    for ns in range(nodes):
        for nd in range(nodes):
            delay = time_idx[ns, 0, nd]   # time_idx checked above
            if not 0 <= delay < horizon:
                raise IndexError()


@cython.boundscheck(False)
@cython.wraparound (False)
def get_state(
        np.ndarray[double, ndim=4, mode="c"] history not None,
        np.ndarray[int, ndim=3, mode="c"] time_idx not None,
        np.ndarray[int, ndim=3] cvar not None,
        np.ndarray[double, ndim=4, mode="c"] delayed_states not None
        ):
    """
    A plain translation from numpy to C loops.
    """
    cdef int nodes, ncvar, modes
    cdef int ns, cv, nd, m
    cdef int delay, cvar_idx
    cdef double h
    nodes = history.shape[2]
    modes = history.shape[3]
    ncvar = cvar.shape[1]

    _common_bounds_check(history, time_idx, cvar, delayed_states)

    # fetch the state
    for ns in range(nodes):
        for cv in range(ncvar):
            cvar_idx = cvar[0, cv, 0]
            for nd in range(nodes):
                delay = time_idx[ns, 0, nd]
                for m in range(modes):
                    h = history[delay, cvar_idx, nd, m]
                    delayed_states[ns, cv, nd, m] = h


@cython.boundscheck(False)
@cython.wraparound (False)
def get_state_with_mask(np.ndarray[double, ndim=4, mode="c"] history not None,
              np.ndarray[int, ndim=3, mode="c"] time_idx not None,
              np.ndarray[int, ndim=3] cvar not None,
              np.ndarray[int, ndim=2, mode="c"] conn_mask not None,
              np.ndarray[double, ndim=4, mode="c"] delayed_states not None
              ):
    """
    Uses a mask to avoid fetching history for uncoupled nodes.
    """
    cdef int nodes, ncvar, modes
    cdef int ns, cv, nd, m
    cdef int delay, cvar_idx
    cdef double h
    cdef int sp
    nodes = history.shape[2]
    modes = history.shape[3]
    ncvar = cvar.shape[1]

    _common_bounds_check(history, time_idx, cvar, delayed_states)
    # bounds check conn_mask[]
    if conn_mask.shape[0] != nodes or conn_mask.shape[1] != nodes:
        raise IndexError()

    for ns in range(nodes):
        for cv in range(ncvar):
            cvar_idx = cvar[0, cv, 0]
            for nd in range(nodes):
                delay = time_idx[ns, 0, nd]
                sp = conn_mask[ns, nd]
                if sp == 0:
                    continue
                for m in range(modes):
                    h = history[delay, cvar_idx, nd, m]
                    delayed_states[ns, cv, nd, m] = h
