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
import threading
import numpy
import pylab
import colorsys
from tvb.basic.config.settings import TVBSettings as config
from tvb.basic.logger.builder import get_logger
from matplotlib import _cntr

#Set the resolution of the phase-plane and sample trajectories.
NUMBEROFGRIDPOINTS = 42
TRAJ_STEPS = 1024


class PhasePlane(object):
    """
    Class responsible with computing the phase plane and trajectories.
    This is view independent.
    """
    def __init__(self, model, integrator):
        self.model = model
        self.integrator = integrator

        self.svx = self.model.state_variables[0] #x-axis: 1st state variable
        if self.model.nvar > 1:
            self.svy = self.model.state_variables[1]  # y-axis: 2nd state variable
        else:
            self.svy = self.model.state_variables[0]
        self.mode = 0
        self.log = get_logger(self.__class__.__module__)


    def _set_state_vector(self):
        """
        Set up a vector containing the default state-variable values and create
        a filler(all zeros) for the coupling arg of the Model's dfun method.
        This method is called once at initialisation (show()).
        """
        #import pdb; pdb.set_trace()
        svr = self.model.state_variable_range
        sv_mean = numpy.array([svr[key].mean() for key in self.model.state_variables])
        sv_mean = sv_mean.reshape((self.model.nvar, 1, 1))
        self.default_sv = sv_mean.repeat(self.model.number_of_modes, axis=2)
        self.no_coupling = numpy.zeros((self.model.nvar, 1, self.model.number_of_modes))


    def _set_mesh_grid(self):
        """
        Generate the phase-plane gridding based on currently selected
        state-variables and their range values.
        """
        xlo = self.model.state_variable_range[self.svx][0]
        xhi = self.model.state_variable_range[self.svx][1]
        ylo = self.model.state_variable_range[self.svy][0]
        yhi = self.model.state_variable_range[self.svy][1]

        self.X = numpy.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS*1j)]
        self.Y = numpy.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS*1j)]


    def _calc_phase_plane(self):
        """ Calculate the vector field. """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        #Calculate the vector field discretely sampled at a grid of points
        grid_point = self.default_sv.copy()
        self.U = numpy.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS, self.model.number_of_modes))
        self.V = numpy.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS, self.model.number_of_modes))
        for ii in xrange(NUMBEROFGRIDPOINTS):
            grid_point[svy_ind] = self.Y[ii]
            for jj in xrange(NUMBEROFGRIDPOINTS):
                #import pdb; pdb.set_trace()
                grid_point[svx_ind] = self.X[jj]
                d = self.model.dfun(grid_point, self.no_coupling)

                for kk in range(self.model.number_of_modes):
                    self.U[ii, jj, kk] = d[svx_ind, 0, kk]
                    self.V[ii, jj, kk] = d[svy_ind, 0, kk]

        #self.UVmag = numpy.sqrt(self.U**2 + self.V**2)
        #import pdb; pdb.set_trace()
        if numpy.isnan(self.U).any() or numpy.isnan(self.V).any():
            self.log.error("NaN")


    def _get_axes_ranges(self, sv):
        lo, hi = self.model.state_variable_range[sv]
        default_range = hi - lo
        min_val = lo - 4.0 * default_range
        max_val = hi + 4.0 * default_range
        return min_val, max_val, lo, hi


    def axis_range(self):
        min_val_x, max_val_x, lo_x, hi_x = self._get_axes_ranges(self.svx)
        min_val_y, max_val_y, lo_y, hi_y = self._get_axes_ranges(self.svy)
        xaxis_range = {'min':min_val_x, 'max':max_val_x, 'lo':lo_x, 'hi':hi_x}
        yaxis_range = {'min':min_val_y, 'max':max_val_y, 'lo':lo_y, 'hi':hi_y}
        return xaxis_range, yaxis_range


    def update_axis(self, mode, svx, svy, x_range, y_range, sv):
        self.mode = mode
        self.svx = svx
        self.svy = svy
        msv_range = self.model.state_variable_range
        msv_range[self.svx][0] = x_range[0]
        msv_range[self.svx][1] = x_range[1]
        msv_range[self.svy][0] = y_range[0]
        msv_range[self.svy][1] = y_range[1]

        for name, val in sv.iteritems():
            k = self.model.state_variables.index(name)
            self.default_sv[k] = val


    def _compute_trajectory(self, x, y):
        """
        Calculate a sample trajectory, starting at the position x,y in the phase-plane.
        """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        state = self.default_sv.copy()
        state[svx_ind] = x
        state[svy_ind] = y
        scheme = self.integrator.scheme
        traj = numpy.zeros((TRAJ_STEPS + 1, self.model.nvar, 1, self.model.number_of_modes))
        traj[0, :] = state

        for step in xrange(TRAJ_STEPS):
            state = scheme(state, self.model.dfun, self.no_coupling, 0.0, 0.0)
            traj[step + 1, :] = state

        return traj



class PhasePlaneInteractive(PhasePlane):
    """
    An interactive phase-plane plot generated from a TVB model object.

    A TVB integrator object will be use for generating sample trajectories
    -- not the phase-plane. This is mainly interesting for visualising
    the effect of noise on a trajectory.
    """


    def __init__(self, model, integrator):
        PhasePlane.__init__(self, model, integrator)
        self.ipp_fig = None
        self.pp_splt = None
        # Concurrent access to the drawing routines is breaking this viewer
        # Concurrency happens despite the GIL because the drawing calls do socket io
        # Lock must be re-entrant because refresh is used in event handling
        self.lock = threading.RLock()


    @staticmethod
    def get_color(num_colours):
        for hue in range(num_colours):
            hue = 1.0 * hue / num_colours
            col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
            yield "#{0:02x}{1:02x}{2:02x}".format(*col)


    def reset(self, model, integrator):
        """
        Resets the state associated with the model and integrator.
        Redraws plot.
        """
        with self.lock:
            PhasePlane.__init__(self, model, integrator)
            self.draw_phase_plane()


    def refresh(self):
        with self.lock:
            self._set_mesh_grid()
            self._calc_phase_plane()
            self._update_phase_plane()


    def draw_phase_plane(self):
        """Generate the interactive phase-plane figure."""
        with self.lock:
            self.log.debug("Plot started...")

            model_name = self.model.__class__.__name__
            msg = "Generating an interactive phase-plane plot for %s"
            self.log.info(msg % model_name)

            self._set_state_vector()

            #TODO: see how we can get the figure size from the UI to better 'fit' the encompassing div
            if self.ipp_fig is None:
                self.ipp_fig = pylab.figure(figsize=(10, 8))
                # add mouse handler for trajectory clicking
                self.ipp_fig.canvas.mpl_connect('button_press_event', self._click_trajectory)

            pylab.clf()
            self.pp_ax = self.ipp_fig.add_axes([0.07, 0.2, 0.9, 0.75])

            self.pp_splt = self.ipp_fig.add_subplot(212)
            self.ipp_fig.subplots_adjust(left=0.07, bottom=0.03, right=0.97, top=0.3, wspace=0.1, hspace=None)
            self.pp_splt.set_color_cycle(self.get_color(self.model.nvar))
            self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                              numpy.zeros((TRAJ_STEPS + 1, self.model.nvar)))
            if hasattr(self.pp_splt, 'autoscale'):
                self.pp_splt.autoscale(enable=True, axis='y', tight=True)
            self.pp_splt.legend(self.model.state_variables)

            #Calculate the phase plane
            self._set_mesh_grid()
            self._calc_phase_plane()

            #Plot phase plane
            self._plot_phase_plane()

            self.ipp_fig.canvas.draw()

            return dict(mplh5ServerURL=config.MPLH5_SERVER_URL, figureNumber=self.ipp_fig.number, showFullToolbar=False)


    ##------------------- Functions for updating the figure ------------------##

    def _update_phase_plane(self):
        """ Clear the axes and redraw the phase-plane. """
        self.pp_ax.clear()
        self.pp_splt.clear()
        self.pp_splt.set_color_cycle(self.get_color(self.model.nvar))
        self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                          numpy.zeros((TRAJ_STEPS + 1, self.model.nvar)))
        if hasattr(self.pp_splt, 'autoscale'):
            self.pp_splt.autoscale(enable=True, axis='y', tight=True)
        self.pp_splt.legend(self.model.state_variables)
        self._plot_phase_plane()


    def _plot_phase_plane(self):
        """ Plot the vector field and its nullclines. """
        # Set title and axis labels
        model_name = self.model.__class__.__name__
        self.pp_ax.set(title=model_name + " mode " + str(self.mode))
        self.pp_ax.set(xlabel="State Variable " + self.svx)
        self.pp_ax.set(ylabel="State Variable " + self.svy)
        #import pdb; pdb.set_trace()
        #Plot a discrete representation of the vector field
        if numpy.all(self.U[:, :, self.mode] + self.V[:, :, self.mode] == 0):
            self.pp_ax.set(title=model_name + " mode " + str(self.mode) + ": NO MOTION IN THIS PLANE")
            X, Y = numpy.meshgrid(self.X, self.Y)
            self.pp_quivers = self.pp_ax.scatter(X, Y, s=8, marker=".", c="k")
        else:
            self.pp_quivers = self.pp_ax.quiver(self.X, self.Y,
                                                self.U[:, :, self.mode],
                                                self.V[:, :, self.mode],
                                                #self.UVmag[:, :, self.mode],
                                                width=0.001, headwidth=8)

        #Plot the nullclines
        self.nullcline_x = self.pp_ax.contour(self.X, self.Y, self.U[:, :, self.mode], [0], colors="r")
        self.nullcline_y = self.pp_ax.contour(self.X, self.Y, self.V[:, :, self.mode], [0], colors="g")
        self.ipp_fig.canvas.draw()


    def _plot_trajectory(self, x, y):
        """
        Plot a sample trajectory, starting at the position x,y in the phase-plane.
        """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        traj = self._compute_trajectory(x, y)

        self.pp_ax.scatter(x, y, s=42, c='g', marker='o', edgecolor=None)
        self.pp_ax.plot(traj[:, svx_ind, 0, self.mode], traj[:, svy_ind, 0, self.mode])

        #Plot the selected state variable trajectories as a function of time
        self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt, traj[:, :, 0, self.mode])

        pylab.draw()


    def _click_trajectory(self, event):
        with self.lock:
            if event.inaxes is self.pp_ax:
                x, y = event.xdata, event.ydata
                self.log.info('trajectory starting at (%f, %f)', x, y)
                self._plot_trajectory(x, y)



class PhasePlaneD3(PhasePlane):
    """
    Provides data for a d3 client
    """

    def __init__(self, model, integrator):
        PhasePlane.__init__(self, model, integrator)

    def compute_phase_plane(self):
        self._set_state_vector()
        self._set_mesh_grid()

        self._calc_phase_plane()
        u = self.U[..., self.mode]
        v = self.V[..., self.mode]
        x, y = numpy.meshgrid(self.X, self.Y)

        d = numpy.dstack((x, y, u, v))
        d = d.reshape((NUMBEROFGRIDPOINTS**2, 4)).tolist()

        xnull = self.nullcline(x, y, u).tolist()
        ynull = self.nullcline(x, y, v).tolist()
        return {'plane': d, 'nullclines': [xnull, ynull]}


    def reset(self, model, integrator):
        PhasePlane.__init__(self, model, integrator)


    # @staticmethod
    def nullcline(self,x, y, z):
        c = _cntr.Cntr(x, y, z)
        # trace a contour
        res = c.trace(0.0)
        if not res:
            return numpy.array([])
        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(res)//2
        segments, codes = res[:nseg], res[nseg:]
        return segments[0]


    def trajectory(self, x, y):
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)
        traj = self._compute_trajectory(x, y)

        signal_x = numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt

        signals = [ zip(signal_x, traj[:, i, 0, self.mode].tolist()) for i in xrange(traj.shape[1])]
        trajectory = zip(traj[:, svx_ind, 0, self.mode], traj[:, svy_ind, 0, self.mode])
        return trajectory, signals
