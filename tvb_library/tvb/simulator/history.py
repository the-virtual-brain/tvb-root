# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
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

"""
Simulator history implementations.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""


import numpy
from tvb.simulator.common import get_logger
from .descriptors import StaticAttr, Dim, NDArray
from .backend.ref import ReferenceBackend

LOG = get_logger(__name__)


class BaseHistory(StaticAttr):
    "Abstract base class for history implementations."

    n_time, n_node, n_cvar, n_mode = Dim(), Dim(), Dim(), Dim()

    weights = NDArray((n_node, n_node), 'f') # type: numpy.ndarray
    delays = NDArray((n_node, n_node), 'f') # type: numpy.ndarray
    cvars = NDArray((n_cvar, ), 'i') # type: numpy.ndarray

    @property
    def nbytes(self):
        arrays = 'weights delays cvars'.split()
        return sum([getattr(self, ary).nbytes for ary in arrays])

    def __init__(self, weights, delays, cvars, n_mode):
        self.n_time, self.n_cvar, self.n_node, self.n_mode = delays.max() + 1, len(cvars), delays.shape[0], n_mode
        self.weights = weights
        self.delays = delays
        self.cvars = cvars

    def initialize(self, init):
        raise NotImplemented

    def update(self, step, new_state):
        raise NotImplemented

    def query(self, step, out=None):
        raise NotImplemented

    @classmethod
    def from_simulator(cls, sim, initial_conditions=None):
        """
        Set initial conditions for the simulation using either the provided
        initial_conditions or, if none are provided, the model's initial()
        method. This method is called durin the Simulator's __init__().

        Any initial_conditions that are provided as an argument are expected
        to have dimensions 1, 2, and 3 with shapse corresponding to the number
        of state_variables, nodes and modes, respectively. If the provided
        inital_conditions are shorter in time (dim=0) than the required history
        the model's initial() method is called to make up the difference.

        """

        backend = ReferenceBackend
        initial_conditions = initial_conditions or sim.initial_conditions
        if initial_conditions is None:
            n_time, n_svar, n_node, n_mode = sim.good_history_shape
            sim.log.info('Preparing initial history of shape %r using model.initial()', sim.good_history_shape)
            if sim.surface is not None:
                n_node = sim.number_of_nodes
            history = sim.model.initial_for_simulator(sim.integrator, (n_time, n_svar, n_node, n_mode))
            initial_conditions = history[n_time - 1]
        # ICs provided
        else:
            # history should be [timepoints, state_variables, nodes, modes]
            sim.log.info('Using provided initial history of shape %r', initial_conditions.shape)
            n_time, n_svar, n_node, n_mode = ic_shape = initial_conditions.shape
            nr = sim.connectivity.number_of_regions
            if sim.surface is not None and n_node == nr:
                initial_conditions = initial_conditions[:, :, sim.surface.region_mapping]
                return sim._configure_history(initial_conditions)
            elif sim.surface is None and ic_shape[1:] != sim.good_history_shape[1:]:
                raise ValueError("Incorrect history sample shape %s, expected %s"
                                 % (ic_shape[1:], sim.good_history_shape[1:]))
            else:
                if ic_shape[0] >= sim.connectivity.horizon:
                    sim.log.debug("Using last %d time-steps for history.", sim.connectivity.horizon)
                    history = initial_conditions[-sim.connectivity.horizon:, :, :, :].copy()
                else:
                    sim.log.debug('Padding initial conditions with model.initial')
                    history = sim.model.initial_for_simulator(sim.integrator, sim.good_history_shape)
                    shift = sim.current_step % sim.connectivity.horizon
                    history = numpy.roll(history, -shift, axis=0)
                    if sim.surface is not None:
                        n_reg = sim.connectivity.number_of_regions
                        (nt, ns, _, nm), ax = history.shape, (2, 0, 1, 3)
                        region_initial_conditions = numpy.zeros((nt, ns, n_reg, nm))
                        backend.add_at(region_initial_conditions.transpose(ax), sim.surface.region_mapping, initial_conditions.transpose(ax))
                        region_initial_conditions /= numpy.bincount(sim.surface.region_mapping).reshape((-1, 1))
                        history[:region_initial_conditions.shape[0], :, :, :] = region_initial_conditions
                    else:
                        history[:ic_shape[0], :, :, :] = initial_conditions
                    history = numpy.roll(history, shift, axis=0)
                sim.current_step += ic_shape[0] - 1

        # Make sure that history values are bounded
        for it in range(history.shape[0]):
            sim.integrator.bound_and_clamp(history[it])
        sim.log.info('Final initial history shape is %r', history.shape)

        # create initial state
        if sim.surface:
            # ensure 4D
            # TODO refactor to backend
            initial_conditions = initial_conditions.reshape((-1, ) + initial_conditions.shape[-3:])
            sim.current_state = initial_conditions[-1].copy()
        else:
            sim.current_state = history[sim.current_step % sim.connectivity.horizon].copy()
        sim.log.info('initial state has shape %r' % (sim.current_state.shape, ))

        # create history buffer
        if sim.surface is not None and history.shape[2] > sim.connectivity.number_of_regions:
            n_reg = sim.connectivity.number_of_regions
            (nt, ns, _, nm), ax = history.shape, (2, 0, 1, 3)
            region_history = numpy.zeros((nt, ns, n_reg, nm))
            backend.add_at(region_history.transpose(ax), sim.surface.region_mapping, history.transpose(ax))
            region_history /= numpy.bincount(sim.surface.region_mapping).reshape((-1, 1))
            history = region_history

        # init history instance
        inst = cls(sim.connectivity.weights, sim.connectivity.idelays,
                   sim.model.cvar, sim.model.number_of_modes)
        inst.initialize(history)
        return inst


class DenseHistory(BaseHistory):
    "TVB's traditional history implementation."

    # extended shape arrays for indexing
    _es = 'n_node', 'n_cvar', 'n_node'
    es_icvar = NDArray(_es, 'i')
    es_idelays = NDArray(_es, 'i')
    es_weights = NDArray(_es + ('n_mode', ), 'f')
    es_node_ids = NDArray(_es, 'i')
    buffer = NDArray(('n_time', 'n_cvar', 'n_node', 'n_mode'), 'f', read_only=False)
    current_state = NDArray(('n_cvar', 'n_node', 'n_mode'), 'f', read_only=False)
    delayed_state = NDArray(('n_node', 'n_cvar', 'n_node', 'n_mode'), 'f', read_only=False)

    @property
    def nbytes(self):
        arrays = 'icvar idelays weights node_ids'.split()
        nbytes = sum([getattr(self, 'es_' + ary).nbytes for ary in arrays])
        nbytes += self.buffer.nbytes
        nbytes += BaseHistory.nbytes.fget(self)
        return nbytes

    def __init__(self, weights, delays, cvars, n_mode):
        super(DenseHistory, self).__init__(weights, delays, cvars, n_mode)

        # initialize indexing arrays
        na = numpy.newaxis
        self.es_icvar = numpy.r_[:len(self.cvars)][na, :, na]
        self.es_idelays = self.delays[:, na, :].astype('i')
        self.es_weights = self.weights[:, na, :, na]
        self.es_node_ids = numpy.r_[:self.n_node][na, na, :]

    def initialize(self, init):
        if init.shape[1] > len(self.cvars):
            init = init[:, self.cvars] # simulator still thinks history is (time, svar, ..)
        self.buffer = init

    def query(self, step, out=None):
        time_idx = (step - 1 - self.es_idelays + self.n_time) % self.n_time
        self.delayed_state = self.buffer[time_idx, self.es_icvar, self.es_node_ids]
        self.current_state = self.buffer[(step - 1) % self.n_time]
        return self.current_state, self.delayed_state

    def update(self, step, new_state):
        self.buffer[step % self.n_time] = new_state[self.cvars]


class SparseHistory(DenseHistory):
    "History implementation which stores data only for non-zero weights."

    n_nnzw = Dim()
    n_nnzr = Dim()
    time_stride = Dim()
    nnz_mask = NDArray(('n_node', 'n_node'), numpy.bool_)
    const_indices = NDArray(('n_cvar', n_nnzw, 'n_mode'), 'i')
    nnz_idelays = NDArray((n_nnzw,), 'i')
    nnz_row_el_idx = NDArray((n_nnzw, ), 'i')
    nnz_col_el_idx = NDArray((n_nnzw, ), 'i')
    nnz_weights = NDArray((n_nnzw, ), 'f')
    nnz_row_idx = NDArray((n_nnzr, ), 'i')

    def __init__(self, weights, delays, cvars, n_mode):
        super(SparseHistory, self).__init__(weights, delays, cvars, n_mode)
        self.time_stride = self.n_cvar * self.n_node * self.n_mode
        self.nnz_mask = weights_nonzero = weights != 0.0 # type: numpy.ndarray
        self.n_nnzw = nnz = weights_nonzero.sum()
        self.nnz_weights = weights[self.nnz_mask]
        self.nnz_row_el_idx, self.nnz_col_el_idx = numpy.argwhere(self.nnz_mask).T
        nnz_row_idx = numpy.unique(self.nnz_row_el_idx)
        self.n_nnzr = len(nnz_row_idx)
        self.nnz_row_idx = nnz_row_idx
        self.nnz_idelays = delays[weights_nonzero].astype('i')
        # build const indices
        n, m = self.n_node, self.n_mode
        icvars_ = numpy.r_[:len(cvars)].reshape((-1, 1, 1)) * n * m
        nodes_ = numpy.tile(numpy.r_[:n], (n, 1))[self.nnz_mask, numpy.newaxis] * m
        modes_ = numpy.r_[:m]
        self.const_indices = icvars_ + nodes_ + modes_
        self.delayed_state[:] = 0.0

        LOG.info('history has n_time=%d n_cvar=%d n_node=%d n_nmode=%d, requires %.2f MB',
                 self.n_time, self.n_cvar, self.n_node, self.n_mode, self.nbytes*2**-20)
        LOG.debug('sparse flat time_stride=%d', self.time_stride)
        LOG.info('sparse history has n_nnzw=%d, i.e. %.2f %% sparse', self.n_nnzw,
                 self.n_nnzw * 100.0 / self.n_node**2)

    def query(self, step, out=None):
        current, delayed = self.query_sparse(step)
        self.delayed_state.transpose((1, 0, 2, 3))[:, self.nnz_mask] = delayed
        return current, self.delayed_state

    def query_sparse(self, step):
        time_indices = ((step - 1 - self.nnz_idelays + self.n_time) % self.n_time) # type: numpy.ndarray
        time_indices = time_indices.reshape((-1, 1)) * self.time_stride # type: numpy.ndarray
        delayed_state = self.buffer.take(time_indices + self.const_indices)
        current_state = self.buffer[(step - 1) % self.n_time]
        return current_state, delayed_state

    @property
    def nbytes(self):
        arrays = 'nnz_mask const_indices nnz_idelays nnz_row_el_idx nnz_col_el_idx nnz_weights nnz_row_idx'.split()
        nbytes = sum([getattr(self, ary).nbytes for ary in arrays])
        nbytes += DenseHistory.nbytes.fget(self)
        return nbytes


# implement in order  NumPy, Numba & OpenCL versions

# simulator.history becomes impl instance

# state must also transpose for performance reasons

# bench history impl like other components

# trace history accesses

# cfun must also now expect to operate on (nnz, ncvar, nmode)