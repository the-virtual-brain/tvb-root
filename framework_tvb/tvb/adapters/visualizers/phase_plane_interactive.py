# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
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
"""
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
from tvb.basic.logger.builder import get_logger
from matplotlib import _cntr

#Set the resolution of the phase-plane and sample trajectories.
NUMBEROFGRIDPOINTS = 42
TRAJ_STEPS = 512


class PhasePlane(object):
    """
    Class responsible with computing the phase plane and trajectories.
    This is view independent.
    """
    def __init__(self, model, integrator):
        self.log = get_logger(self.__class__.__module__)
        self.model = model
        self.integrator = integrator

        self.mode = 0
        self.svx_ind = 0    # x-axis: 1st state variable
        if self.model.nvar > 1:
            self.svy_ind = 1  # y-axis: 2nd state variable
        else:
            self.svy_ind = 0
        self._set_state_vector()
        self._create_mesh_jitter()


    def _set_state_vector(self):
        """
        Set up a vector containing the default state-variable values and create
        a filler(all zeros) for the coupling arg of the Model's dfun method.
        This method is called once at initialisation (show()).
        """
        svr = self.model.state_variable_range
        sv_mean = numpy.array([svr[key].mean() for key in self.model.state_variables])
        sv_mean = sv_mean.reshape((self.model.nvar, 1, 1))
        self.default_sv = sv_mean.repeat(self.model.number_of_modes, axis=2)
        self.no_coupling = numpy.zeros((self.model.nvar, 1, self.model.number_of_modes))


    def _create_mesh_jitter(self):
        shape = NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS
        d =  1.0 / (4 * NUMBEROFGRIDPOINTS)
        self._jitter =  numpy.random.normal(0, d, shape), numpy.random.normal(0, d, shape)


    def _get_mesh_grid(self, noisy=True):
        """
        Generate the phase-plane gridding based on currently selected
        state-variables and their range values.
        """
        svr = self.model.state_variable_range
        xlo, xhi = svr[self.model.state_variables[self.svx_ind]]
        ylo, yhi = svr[self.model.state_variables[self.svy_ind]]

        xg = numpy.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS * 1j)]
        yg = numpy.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS * 1j)]
        xgr, ygr = numpy.meshgrid(xg, yg)
        if noisy:
            # add scaled noise
            xgr += self._jitter[0] * (xhi - xlo)
            ygr += self._jitter[1] * (yhi - ylo)
        return xgr, ygr


    def _calc_phase_plane(self, xg, yg):
        """
        Computes the vector field. It takes a x, y coordinate mesh and returns a u, v vector field.
        Vectorized function, it evaluate all grid points at once as if they were connectivity nodes
        """
        state_variables = numpy.tile(self.default_sv, (NUMBEROFGRIDPOINTS ** 2, 1))

        for mode_idx in xrange(self.model.number_of_modes):
            state_variables[self.svx_ind, :, mode_idx] = xg.flat
            state_variables[self.svy_ind, :, mode_idx] = yg.flat

        d_grid = self.model.dfun(state_variables, self.no_coupling)

        flat_uv_grid = d_grid[[self.svx_ind, self.svy_ind], :, :]  # subset of the state variables to be displayed
        u, v = flat_uv_grid.reshape((2, NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS, self.model.number_of_modes))
        if numpy.isnan(u).any() or numpy.isnan(v).any():
            self.log.error("NaN")
        return u, v


    def get_axes_ranges(self, sv):
        lo, hi = self.model.state_variable_range[sv]
        default_range = hi - lo
        min_val = lo - 4.0 * default_range
        max_val = hi + 4.0 * default_range
        return min_val, max_val, lo, hi


    def update_axis(self, mode, svx, svy, x_range, y_range, sv):
        self.mode = mode
        self.svx_ind = self.model.state_variables.index(svx)
        self.svy_ind = self.model.state_variables.index(svy)
        svr = self.model.state_variable_range
        svr[svx][:] = x_range #todo is it ok to update the model?
        svr[svy][:] = y_range

        for name, val in sv.iteritems():
            k = self.model.state_variables.index(name)
            self.default_sv[k] = val


    def _compute_trajectories(self, states):
        """
        A vectorized method of computing a number of trajectories in parallel.
        The trajectories will be projected on the plane defined by svx_ind, svy_ind.
        """
        scheme = self.integrator.scheme
        trajs = numpy.zeros((TRAJ_STEPS + 1, self.model.nvar, len(states), self.model.number_of_modes))
        # reshape to what dfun expects: from n, sv to sv, n, mode
        states = numpy.tile(states.T[:, :, numpy.newaxis], self.model.number_of_modes)
        trajs[0, :] = states
        # grow trajectories step by step
        for step in xrange(TRAJ_STEPS):
            states = scheme(states, self.model.dfun, self.no_coupling, 0.0, 0.0)
            trajs[step + 1, :] = states

        if numpy.isnan(trajs).any():
            self.log.warn("NaN in trajectories")
        return trajs



class PhasePlaneD3(PhasePlane):
    """
    Provides data for a d3 client
    """

    def compute_phase_plane(self):
        """
        :return: A json representation of the phase plane.
        """
        x, y = self._get_mesh_grid(noisy=True)

        u, v = self._calc_phase_plane(x, y)
        u = u[..., self.mode]
        v = v[..., self.mode]

        d = numpy.dstack((x, y, u, v))
        d = d.reshape((NUMBEROFGRIDPOINTS ** 2, 4)).tolist()

        xnull = [{'path': segment.tolist(), 'nullcline_index': 0} for segment in self.nullcline(x, y, u)]
        ynull = [{'path': segment.tolist(), 'nullcline_index': 1} for segment in self.nullcline(x, y, v)]
        return {'plane': d, 'nullclines': xnull + ynull}


    # @staticmethod
    def nullcline(self, x, y, z):
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


    def _state_dict_to_array(self, state):
        arr = numpy.zeros(len(self.model.state_variables))
        for svn, v in state.iteritems():
            svn_idx = self.model.state_variables.index(svn)
            arr[svn_idx] = v
        return arr


    def trajectories(self, starting_points):
        """
        :param starting_points: A list of starting points represented as dicts of state_var_name to value
        :return: a tuple of trajectories and signals
        """
        starting_points = numpy.array([self._state_dict_to_array(s) for s in starting_points])
        traj = self._compute_trajectories(starting_points) # point_on_traj_idx, sv_idx, traj_idx, mode
        # reshape it and project it on the plane defined  by the current axis state vars
        traj = traj.transpose(2, 0, 1, 3) # traj_idx, point, sv_idx, mode
        trajectory = traj[:, :, [self.svx_ind, self.svy_ind], self.mode] # traj_idx, points, x, y

        # signals for last trajectory
        signal_x = numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt
        signals = [ zip(signal_x, traj[-1, :, i, self.mode].tolist()) for i in xrange(self.model.nvar)]

        return trajectory.tolist(), signals
