# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
"""
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
import six
from tvb.basic.logger.builder import get_logger

# To plot nullclines we need a function that computes contours of a scalar field.
# We use the internal matplotlib one.
# Now that newer matplotlib versions have changed this internal API it would be a
# good idea to use a proper library for this.

try:
    from matplotlib import _cntr
    # older matplotlib

    def nullcline(x, y, z):
        c = _cntr.Cntr(x, y, z)
        # trace a contour
        res = c.trace(0.0)
        if not res:
            return numpy.array([])
        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(res) // 2
        segments, codes = res[:nseg], res[nseg:]
        return segments

except ImportError:
    from matplotlib import _contour
    # newer matplotlib >= 2.2

    def nullcline(x, y, z):
        c = _contour.QuadContourGenerator(x, y, z, None, True, 0)
        segments = c.create_contour(0.0)
        return segments[0]

# how much courser is the grid used to show the vectors
GRID_SUBSAMPLE = 2
# Set the resolution of the phase-plane and sample trajectories.
NUMBEROFGRIDPOINTS = GRID_SUBSAMPLE * 42


class _PhaseSpace(object):
    """
    Dimensionality independent code
    """

    def __init__(self, model, integrator):
        self.log = get_logger(self.__class__.__module__)
        self.model = model
        self.integrator = integrator

    def _compute_trajectories(self, states, n_steps):
        """
        A vectorized method of computing a number of trajectories in parallel.
        Returns a collection of nvar-dimensional trajectories.
        """
        scheme = self.integrator.scheme
        trajs = numpy.zeros((n_steps + 1, self.model.nvar, len(states), self.model.number_of_modes))
        # reshape to what dfun expects: from n, sv to sv, n, mode
        states = numpy.tile(states.T[:, :, numpy.newaxis], self.model.number_of_modes)
        trajs[0, :] = states
        no_coupling = numpy.zeros((self.model.nvar, states.shape[1], self.model.number_of_modes))
        # grow trajectories step by step
        for step in range(n_steps):
            states = scheme(states, self.model.dfun, no_coupling, 0.0, 0.0)
            trajs[step + 1, :] = states

        if numpy.isnan(trajs).any():
            self.log.warning("NaN in trajectories")
        return trajs

    def get_axes_ranges(self, sv):
        lo, hi = self.model.state_variable_range[sv]
        default_range = hi - lo
        min_val = lo - 4.0 * default_range
        max_val = hi + 4.0 * default_range
        return min_val, max_val, lo, hi


class PhasePlane(_PhaseSpace):
    """
    Responsible with computing phase space slices and trajectories.
    A collection of math-y utilities it is view independent (holds no state related to views).
    """

    @staticmethod
    def _create_mesh_jitter():
        shape = NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS
        d = 1.0 / (4 * NUMBEROFGRIDPOINTS)
        return numpy.random.normal(0, d, shape), numpy.random.normal(0, d, shape)

    @staticmethod
    def _get_mesh_grid(x_range, y_range, noise=None):
        """
        Generate the phase-plane gridding based on the given
        state-variable indices and their range values.
        """
        xlo, xhi = x_range
        ylo, yhi = y_range

        xg = numpy.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS * 1j)]
        yg = numpy.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS * 1j)]
        xgr, ygr = numpy.meshgrid(xg, yg)
        if noise:
            # add scaled noise
            xgr += noise[0] * (xhi - xlo)
            ygr += noise[1] * (yhi - ylo)
        return xgr, ygr

    def _calc_phase_plane(self, state, svx_ind, svy_ind, xg, yg):
        """
        Computes a 2d axis aligned rectangle of the vector field returning a u, v vector field.
        The slice passes through the `state` point and varies along the axes given by svx_ind and svy_ind.
        The last 2 parameters specify the mesh in the varying directions. To be computed by _get_mesh_grid
        Vectorized function, it evaluates all grid points at once as if they were connectivity nodes.
        """
        state_variables = numpy.tile(state, (NUMBEROFGRIDPOINTS ** 2, 1))

        for mode_idx in range(self.model.number_of_modes):
            state_variables[svx_ind, :, mode_idx] = xg.flat
            state_variables[svy_ind, :, mode_idx] = yg.flat

        no_coupling = numpy.zeros((self.model.nvar, state_variables.shape[1], self.model.number_of_modes))
        d_grid = self.model.dfun(state_variables, no_coupling)

        flat_uv_grid = d_grid[[svx_ind, svy_ind], :, :]  # subset of the state variables to be displayed
        u, v = flat_uv_grid.reshape((2, NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS, self.model.number_of_modes))
        if numpy.isnan(u).any() or numpy.isnan(v).any():
            self.log.error("NaN")
        return u, v


class PhasePlaneD3(PhasePlane):
    """
    Provides data for a d3 client
    """

    def __init__(self, model, integrator):
        PhasePlane.__init__(self, model, integrator)
        self.mode = 0
        self.svx_ind = 0  # x-axis: 1st state variable
        if self.model.nvar > 1:
            self.svy_ind = 1  # y-axis: 2nd state variable
        else:
            self.svy_ind = 0
        self.x_range = None
        self.y_range = None
        # Set up a vector containing the default state-variable values
        svr = self.model.state_variable_range
        sv_mean = numpy.array([svr[key].mean() for key in self.model.state_variables])
        sv_mean = sv_mean.reshape((self.model.nvar, 1, 1))
        self.default_sv = sv_mean.repeat(self.model.number_of_modes, axis=2)
        self.update_integrator_clamping()
        self._jitter = None  # self._create_mesh_jitter()

    def update_integrator_clamping(self):
        clamped_sv_indices = [i for i in range(self.model.nvar) if i not in [self.svx_ind, self.svy_ind]]
        if clamped_sv_indices:
            self.integrator.clamped_state_variable_indices = numpy.array(clamped_sv_indices)
            self.integrator.clamped_state_variable_values = self.default_sv[
                self.integrator.clamped_state_variable_indices]
        else:
            self.integrator.clamped_state_variable_indices = None
            self.integrator.clamped_state_variable_values = None

    def update_axis(self, mode, svx, svy, x_range, y_range, state_vars):
        self.mode = int(mode)
        self.svx_ind = self.model.state_variables.index(svx)
        self.svy_ind = self.model.state_variables.index(svy)
        self.x_range = x_range
        self.y_range = y_range

        for name, val in six.iteritems(state_vars):
            k = self.model.state_variables.index(name)
            self.default_sv[k] = val
        self.update_integrator_clamping()

    def compute_phase_plane(self):
        """
        :return: A json representation of the phase plane.
        """
        x, y = self._get_mesh_grid(self.x_range, self.y_range, noise=self._jitter)

        u, v = self._calc_phase_plane(self.default_sv, self.svx_ind, self.svy_ind, x, y)
        u = u[..., self.mode]  # project on active mode
        v = v[..., self.mode]
        xnull = [{'path': segment.tolist(), 'nullcline_index': 0} for segment in nullcline(x, y, u)]
        ynull = [{'path': segment.tolist(), 'nullcline_index': 1} for segment in nullcline(x, y, v)]

        # a courser mesh for the arrows
        xsmall = x[::GRID_SUBSAMPLE, ::GRID_SUBSAMPLE]
        ysmall = y[::GRID_SUBSAMPLE, ::GRID_SUBSAMPLE]
        usmall = u[::GRID_SUBSAMPLE, ::GRID_SUBSAMPLE]
        vsmall = v[::GRID_SUBSAMPLE, ::GRID_SUBSAMPLE]

        d = numpy.dstack((xsmall, ysmall, usmall, vsmall))
        d = d.reshape(((NUMBEROFGRIDPOINTS // GRID_SUBSAMPLE) ** 2, 4)).tolist()

        return {'plane': d, 'nullclines': xnull + ynull}

    def _state_dict_to_array(self, state):
        arr = numpy.zeros(len(self.model.state_variables))
        for svn, v in six.iteritems(state):
            svn_idx = self.model.state_variables.index(svn)
            arr[svn_idx] = v
        return arr

    def trajectories(self, starting_points, n_steps=512):
        """
        :param starting_points: A list of starting points represented as dicts of state_var_name to value
        :return: a tuple of trajectories and signals
        """
        starting_points = numpy.array([self._state_dict_to_array(s) for s in starting_points])
        traj = self._compute_trajectories(starting_points, n_steps)  # point_on_traj_idx, sv_idx, traj_idx, mode
        # reshape it and project it on the plane defined  by the current axis state vars
        traj = traj.transpose(2, 0, 1, 3)  # traj_idx, point, sv_idx, mode
        trajectory = traj[:, :, [self.svx_ind, self.svy_ind], self.mode]  # traj_idx, points, x, y

        # signals for last trajectory
        signal_x = numpy.arange(n_steps + 1) * self.integrator.dt
        signals = [list(zip(signal_x, traj[-1, :, i, self.mode].tolist())) for i in [self.svx_ind, self.svy_ind]]

        return trajectory.tolist(), signals


class PhaseLineD3(_PhaseSpace):
    def __init__(self, model, integrator):
        _PhaseSpace.__init__(self, model, integrator)
        self.mode = 0

    def _grid(self):
        svr = self.model.state_variable_range
        xlo, xhi = svr[self.model.state_variables[0]]
        return numpy.linspace(xlo, xhi, NUMBEROFGRIDPOINTS)

    def compute_phase_plane(self):
        xg = self._grid()
        # dfun modifies state in place, so we need to copy xg
        state = xg.reshape((1, NUMBEROFGRIDPOINTS, 1)).copy()  # will broadcast to modes
        no_coupling = numpy.zeros((self.model.nvar, state.shape[1], self.model.number_of_modes))
        u = self.model.dfun(state, no_coupling)
        u = u[0, :, self.mode]

        d = numpy.vstack((xg, u)).T
        if numpy.isnan(d).any():
            self.log.error("NaN")

        # find zeroes. This method is not exact
        zero_crossings = numpy.where(numpy.diff(numpy.sign(u)))[0]
        zero_crossings = xg[zero_crossings]
        return {'signal': d.tolist(), 'zeroes': zero_crossings.tolist()}

    def update_axis(self, mode, svx, x_range):
        self.mode = int(mode)
        svr = self.model.state_variable_range
        svr[svx][:] = x_range


def phase_space_d3(model, integrator):
    """
    :return: A phase plane or a phase line depending on the dimensionality of the model
    """
    if model.nvar == 1:
        return PhaseLineD3(model, integrator)
    else:
        return PhasePlaneD3(model, integrator)
