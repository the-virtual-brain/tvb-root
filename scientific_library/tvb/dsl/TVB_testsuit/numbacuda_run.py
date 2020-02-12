#from __future__ import division, print_function
import math as m
import numpy as _lpy_np
import numba.cuda as _lpy_ncu
import numba as _lpy_numba
from tvb_hpc import utils, network
from typing import List

import logging

LOG = utils.getLogger('tvb_hpc')


@_lpy_ncu.jit(
    "void(int64, int64, int64, float32[::1], float32[::1], float32[::1], float32[::1], float32[::1], float32[::1], int64, uint32[::1], uint32[::1], uint32[::1] , float32[::1], float32[::1], int64)")
def Kuramoto_and_Network_and_EulerStep_inner(nstep, nnode, ntime, state, input, param, drift, _diffs, obsrv, nnz,
                                             delays, row, col, weights, a, i_step_0):
    # Get the id for each thread
    tcoupling = _lpy_ncu.threadIdx.x
    tspeed = _lpy_ncu.blockIdx.x
    sid = _lpy_ncu.gridDim.x
    idp = tspeed * _lpy_ncu.blockDim.x + tcoupling
    # for each simulation step and for each node in the system
    for i_step in range(0, nstep):
        step = (idp * ntime + ((i_step + i_step_0) % ntime)) * nnode
        for i_node in range(0, nnode):
            # calculate the node index
            idx = idp * nnode + i_node
            # get the node params, in this case only omega
            omega = param[i_node]
            # retrieve the range of connected nodes
            j_node_lo = row[i_node]
            j_node_hi = row[i_node + 1]
            # calculate the input from other nodes at the current step
            acc_j_node = 0.0
            theta_i = obsrv[(step + i_node) * 2]
            for j_node in range(j_node_lo, j_node_hi):
                dij = delays[tspeed * nnz + j_node] * nnode
                column = col[j_node]  # ONE MORE THAN C
                # column = 0
                wij = weights[j_node]
                theta_j = obsrv[(step - dij + column) * 2]
                acc_j_node += wij * m.sin(theta_j - theta_i)
            input_tmp = a[tcoupling] * acc_j_node / nnode
            # calculate the whole drift for the simulation step
            drift_tmp = omega + input_tmp
            # update the state
            state_tmp = state[idx] + drift_tmp
            # wrap the state within the desired limits
            # TODO: Should this be an half-open interval?!
            state_tmp += (state_tmp < 0) * 2 * _lpy_np.pi
            state_tmp -= (state_tmp > 2 * _lpy_np.pi) * 2 * _lpy_np.pi
            # write the state to the observables data structure
            obsrv[(step + i_node) * 2 + 1] = m.sin(state_tmp)
            obsrv[(step + i_node) * 2] = state_tmp
            input[idx] = input_tmp
            drift[idx] = drift_tmp
            state[idx] = state_tmp


class NumbaCudaRun:

    def __init__(self):
        return

    def prep_arrays(self, nsims, nnode: int) -> List[_lpy_np.ndarray]:
        """
        Prepare arrays for use with this model.
        """
        dtype = _lpy_np.float32
        arrs: List[_lpy_np.ndarray] = []
        for key in 'input drift diffs'.split():
            shape = nsims * nnode * 1
            arrs.append(_lpy_np.zeros(shape, dtype))
        for i, (lo, hi) in enumerate([(0, 2 * _lpy_np.pi)]):
            state = _lpy_np.ones(nsims * nnode, dtype="float32")  # .random.uniform(float(lo), float(hi),
        arrs.append(state)
        param = _lpy_np.ones((nnode * 1), dtype)
        arrs.append(param)
        return arrs

    def run_all(self, args):
        j, speed, coupling, nnode, lengths, nz, nnz, row, col, wnz, dt = args
        lnz = []
        for i in range(len(speed)):
            lnz.append((lengths[nz] / speed[i] / dt).astype(_lpy_np.uintc))
        flat_lnz = _lpy_np.reshape(lnz, (nnz * len(speed)))
        input, drift, diffs, state, param = prep_arrays(len(coupling) * len(speed), nnode)
        obsrv = _lpy_np.zeros((len(coupling) * len(speed) * (max(flat_lnz) + 3 + 4000) * nnode * 2), _lpy_np.float32)
        trace = _lpy_np.zeros((len(coupling) * len(speed), 400, nnode), _lpy_np.float32)
        threadsperblock = len(coupling)
        blockspergrid = len(speed)
        for i in range(400):
            Kuramoto_and_Network_and_EulerStep_inner[blockspergrid, threadsperblock](10, nnode,
                                                                                     (max(flat_lnz) + 3 + 4000), state,
                                                                                     input, param, drift, diffs, obsrv,
                                                                                     nnz, flat_lnz, row, col, wnz,
                                                                                     coupling, i * 10)
            o = obsrv
            o = _lpy_np.reshape(o, (len(coupling) * len(speed), (max(flat_lnz) + 3 + 4000), nnode, 2))
            trace[:, i, :] = o[:, i * 10:(i + 1) * 10, :, 0].sum(axis=1)
        print(Kuramoto_and_Network_and_EulerStep_inner.inspect_types())
        return trace

    def run_simulation(self, couplings, speeds, dt, sim_length, connectivity):
        _lpy_ncu.select_device(0)
        logging.getLogger("numba").setLevel(logging.WARNING)
        LOG.info(_lpy_ncu.gpus)
        lengths = connectivity.lengths
        nnz = connectivity.nnz
        col = connectivity.col
        wnz = connectivity.weights
        nz = connectivity.nz
        nnode = 68
        # Make parallel over speed anc coupling
        trace = run_all((0, speeds, couplings, nnode, lengths, nz, nnz, row, col, wnz))
        return trace
