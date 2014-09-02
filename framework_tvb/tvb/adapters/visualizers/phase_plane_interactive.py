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
from matplotlib.widgets import Slider, Button, RadioButtons
from tvb.basic.config.settings import TVBSettings as config
from tvb.basic.logger.builder import get_logger


# Define a colour theme... see: matplotlib.colors.cnames.keys()
AXCOLOUR = "steelblue"  # 'burlywood'
BUTTONCOLOUR = "steelblue"  # 'fuchsia'
HOVERCOLOUR = "darkred"  # 'chartreuse'

#Set the resolution of the phase-plane and sample trajectories.
NUMBEROFGRIDPOINTS = 42
TRAJ_STEPS = 1024


def get_color(num_colours):
    for hue in range(num_colours):
        hue = 1.0 * hue / num_colours
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)



class PhasePlaneInteractive(object):
    """
    An interactive phase-plane plot generated from a TVB model object.

    A TVB integrator object will be use for generating sample trajectories
    -- not the phase-plane. This is mainly interesting for visualising
    the effect of noise on a trajectory.
    """


    def __init__(self, model, integrator):
        self.log = get_logger(self.__class__.__module__)
        self.model = model
        self.integrator = integrator

        self.ipp_fig = None
        self.pp_splt = None
        # Concurrent access to the drawing routines is breaking this viewer
        # Concurrency happens despite the GIL because the drawing calls do socket io
        # Lock must be re-entrant because refresh is used in event handling
        self.lock = threading.RLock()


    def reset(self, model, integrator):
        """
        Resets the state associated with the model and integrator.
        Redraws plot.
        """
        with self.lock:
            self.model = model
            self.integrator = integrator
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

            self.svx = self.model.state_variables[0]  # x-axis: 1st state variable
            if self.model.nvar > 1:
                self.svy = self.model.state_variables[1]  # y-axis: 2nd state variable
            else:
                self.svy = self.model.state_variables[0]
            self.mode = 0

            self._set_state_vector()

            #TODO: see how we can get the figure size from the UI to better 'fit' the encompassing div
            if self.ipp_fig is None:
                self.ipp_fig = pylab.figure(figsize=(10, 8))
                # add mouse handler for trajectory clicking
                self.ipp_fig.canvas.mpl_connect('button_press_event', self._click_trajectory)

            pylab.clf()
            self.pp_ax = self.ipp_fig.add_axes([0.265, 0.2, 0.7, 0.75])

            self.pp_splt = self.ipp_fig.add_subplot(212)
            self.ipp_fig.subplots_adjust(left=0.265, bottom=0.02, right=0.75, top=0.3, wspace=0.1, hspace=None)
            self.pp_splt.set_color_cycle(get_color(self.model.nvar))
            self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                              numpy.zeros((TRAJ_STEPS + 1, self.model.nvar)))
            if hasattr(self.pp_splt, 'autoscale'):
                self.pp_splt.autoscale(enable=True, axis='y', tight=True)
            self.pp_splt.legend(self.model.state_variables)

            #Selectors
            self._add_state_variable_selector()
            self._add_mode_selector()

            #Sliders
            self._add_axes_range_sliders()
            self._add_state_variable_sliders()

            #Reset buttons
            #self._add_reset_param_button()
            self._add_reset_sv_button()
            self._add_reset_axes_button()

            #Calculate the phase plane
            self._set_mesh_grid()
            self._calc_phase_plane()

            #Plot phase plane
            self._plot_phase_plane()

            self.ipp_fig.canvas.draw()

            return dict(mplh5ServerURL=config.MPLH5_SERVER_URL, figureNumber=self.ipp_fig.number, showFullToolbar=False)


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


    def _add_state_variable_selector(self):
        """
        Generate radio selector buttons to set which state variable is
        displayed on the x and y axis of the phase-plane plot.
        """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        #State variable for the x axis
        pos_shp = [0.08, 0.07, 0.065, 0.12 + 0.006 * self.model.nvar]
        rax = self.ipp_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="x-axis")
        self.state_variable_x = RadioButtons(rax, tuple(self.model.state_variables), active=svx_ind)
        self.state_variable_x.on_clicked(self._update_svx)

        #State variable for the y axis
        pos_shp = [0.16, 0.07, 0.065, 0.12 + 0.006 * self.model.nvar]
        rax = self.ipp_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="y-axis")
        self.state_variable_y = RadioButtons(rax, tuple(self.model.state_variables), active=svy_ind)
        self.state_variable_y.on_clicked(self._update_svy)


    def _add_mode_selector(self):
        """
        Add a radio button to the figure for selecting which mode of the model
        should be displayed.
        """
        pos_shp = [0.02, 0.07, 0.04, 0.1 + 0.002 * self.model.number_of_modes]
        rax = self.ipp_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="Mode")
        mode_tuple = tuple(range(self.model.number_of_modes))
        self.mode_selector = RadioButtons(rax, mode_tuple, active=0)
        self.mode_selector.on_clicked(self._update_mode)


    def _add_axes_range_sliders(self):
        """
        Add sliders to the figure to allow the phase-planes axes to be set.
        """
        self.axes_range_sliders = dict()

        state_var_x_range = self.model.state_variable_range[self.svx]
        state_var_y_range = self.model.state_variable_range[self.svy]
        default_range_x = state_var_x_range[1] - state_var_x_range[0]
        default_range_y = state_var_y_range[1] - state_var_y_range[0]
        min_val_x = state_var_x_range[0] - 4.0 * default_range_x
        max_val_x = state_var_x_range[1] + 4.0 * default_range_x
        min_val_y = state_var_y_range[0] - 4.0 * default_range_y
        max_val_y = state_var_y_range[1] + 4.0 * default_range_y

        sax = self.ipp_fig.add_axes([0.04, 0.835, 0.125, 0.025], axisbg=AXCOLOUR)
        sl_x_min = Slider(sax, "xlo", min_val_x, max_val_x, valinit=state_var_x_range[0])
        sl_x_min.on_changed(self._update_range)

        sax = self.ipp_fig.add_axes([0.04, 0.8, 0.125, 0.025], axisbg=AXCOLOUR)
        sl_x_max = Slider(sax, "xhi", min_val_x, max_val_x, valinit=state_var_x_range[1])
        sl_x_max.on_changed(self._update_range)

        sax = self.ipp_fig.add_axes([0.04, 0.765, 0.125, 0.025], axisbg=AXCOLOUR)
        sl_y_min = Slider(sax, "ylo", min_val_y, max_val_y, valinit=state_var_y_range[0])
        sl_y_min.on_changed(self._update_range)

        sax = self.ipp_fig.add_axes([0.04, 0.73, 0.125, 0.025], axisbg=AXCOLOUR)
        sl_y_max = Slider(sax, "yhi", min_val_y, max_val_y, valinit=state_var_y_range[1])
        sl_y_max.on_changed(self._update_range)

        self.axes_range_sliders["sl_x_min"] = sl_x_min
        self.axes_range_sliders["sl_x_max"] = sl_x_max
        self.axes_range_sliders["sl_y_min"] = sl_y_min
        self.axes_range_sliders["sl_y_max"] = sl_y_max


    def _add_state_variable_sliders(self):
        """
        Add sliders to the figure to allow default values for the models state
        variable to be set.
        """
        msv_range = self.model.state_variable_range
        offset = 0.0
        self.sv_sliders = dict()
        for sv in range(self.model.nvar):
            offset += 0.035
            pos_shp = [0.04, 0.6 - offset, 0.125, 0.025]
            sax = self.ipp_fig.add_axes(pos_shp, axisbg=AXCOLOUR)
            sv_str = self.model.state_variables[sv]
            self.sv_sliders[sv_str] = Slider(sax, sv_str, msv_range[sv_str][0],
                                             msv_range[sv_str][1], valinit=self.default_sv[sv, 0, 0])
            self.sv_sliders[sv_str].on_changed(self._update_state_variables)


    def _add_reset_sv_button(self):
        """
        Add a button to the figure for reseting the model state variables to
        their default values.
        """
        bax = self.ipp_fig.add_axes([0.04, 0.60, 0.125, 0.04])
        self.reset_sv_button = Button(bax, 'Reset state-variables', color=BUTTONCOLOUR, hovercolor=HOVERCOLOUR)

        def reset_state_variables(event):
            for svsl in self.sv_sliders.itervalues():
                svsl.reset()

        self.reset_sv_button.on_clicked(reset_state_variables)


    def _add_reset_axes_button(self):
        """
        Add a button to the figure for resetting the phase-plane axes to their
        default ranges.
        """
        bax = self.ipp_fig.add_axes([0.04, 0.87, 0.125, 0.04])
        self.reset_axes_button = Button(bax, 'Reset axes', color=BUTTONCOLOUR, hovercolor=HOVERCOLOUR)

        def reset_ranges(event):
            self.axes_range_sliders["sl_x_min"].reset()
            self.axes_range_sliders["sl_x_max"].reset()
            self.axes_range_sliders["sl_y_min"].reset()
            self.axes_range_sliders["sl_y_max"].reset()

        self.reset_axes_button.on_clicked(reset_ranges)


    ##------------------------------------------------------------------------##
    ##------------------- Functions for updating the figure ------------------##
    ##------------------------------------------------------------------------##

    #NOTE: All the ax.set_xlim, poly.xy, etc, garbage below is fragile. It works 
    #      at the moment, but there are currently bugs in Slider and the hackery
    #      below takes these into account... If the bugs are fixed/changed then
    #      this could break. As an example, the Slider doc says poly is a
    #      Rectangle, but it's actually a Polygon. The Slider set_val method 
    #      assumes a Rectangle even though this is not the case, so the array 
    #      Slider.poly.xy is corrupted by that method. The corruption isn't 
    #      visible in the plot, which is probably why it hasn't been fixed...

    def update_range_sliders(self, sv, sl_min_key, sl_max_key):
        state_var_range = self.model.state_variable_range[sv]
        default_range = state_var_range[1] - state_var_range[0]
        min_val = state_var_range[0] - 4.0 * default_range
        max_val = state_var_range[1] + 4.0 * default_range

        sl_min = self.axes_range_sliders[sl_min_key]
        sl_max = self.axes_range_sliders[sl_max_key]

        sl_min.valinit = state_var_range[0]
        sl_min.valmin = min_val
        sl_min.valmax = max_val
        sl_min.ax.set_xlim(min_val, max_val)
        sl_min.poly.axes.set_xlim(min_val, max_val)
        sl_min.poly.xy[[0, 1], 0] = min_val
        sl_min.vline.set_data(([sl_min.valinit, sl_min.valinit], [0, 1]))

        sl_max.valinit = state_var_range[1]
        sl_max.valmin = min_val
        sl_max.valmax = max_val
        sl_max.ax.set_xlim(min_val, max_val)
        sl_max.poly.axes.set_xlim(min_val, max_val)
        sl_max.poly.xy[[0, 1], 0] = min_val
        sl_max.vline.set_data(([sl_max.valinit, sl_max.valinit], [0, 1]))

        sl_min.reset()
        sl_max.reset()


    def _update_svx(self, label):
        """
        Update state variable used for x-axis based on radio buttton selection.
        """
        self.svx = label
        self.update_range_sliders(self.svx, "sl_x_min", "sl_x_max")
        self.refresh()


    def _update_svy(self, label):
        """
        Update state variable used for y-axis based on radio buttton selection.
        """
        self.svy = label
        self.update_range_sliders(self.svy, "sl_y_min", "sl_y_max")
        self.refresh()


    def _update_mode(self, label):
        """ Update the visualised mode based on radio button selection. """
        self.mode = label
        self._update_phase_plane()


    def _update_range(self, val):
        """
        Update the axes ranges based on the current axes slider values.

        NOTE: Haven't figured out how to update independently, so just update everything.
        """
        #TODO: Grab caller and use val directly, ie independent range update.
        sl_x_min = self.axes_range_sliders["sl_x_min"]
        sl_x_max = self.axes_range_sliders["sl_x_max"]
        sl_y_min = self.axes_range_sliders["sl_y_min"]
        sl_y_max = self.axes_range_sliders["sl_y_max"]

        sl_x_min.ax.set_axis_bgcolor(AXCOLOUR)
        sl_x_max.ax.set_axis_bgcolor(AXCOLOUR)
        sl_y_min.ax.set_axis_bgcolor(AXCOLOUR)
        sl_y_max.ax.set_axis_bgcolor(AXCOLOUR)

        if sl_x_min.val >= sl_x_max.val:
            self.log.error("X-axis min must be less than max...")
            sl_x_min.ax.set_axis_bgcolor("Red")
            sl_x_max.ax.set_axis_bgcolor("Red")
            return

        if sl_y_min.val >= sl_y_max.val:
            self.log.error("Y-axis min must be less than max...")
            sl_y_min.ax.set_axis_bgcolor("Red")
            sl_y_max.ax.set_axis_bgcolor("Red")
            return

        msv_range = self.model.state_variable_range
        msv_range[self.svx][0] = sl_x_min.val
        msv_range[self.svx][1] = sl_x_max.val
        msv_range[self.svy][0] = sl_y_min.val
        msv_range[self.svy][1] = sl_y_max.val

        self.refresh()


    def _update_phase_plane(self):
        """ Clear the axes and redraw the phase-plane. """
        self.pp_ax.clear()
        self.pp_splt.clear()
        self.pp_splt.set_color_cycle(get_color(self.model.nvar))
        self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                          numpy.zeros((TRAJ_STEPS + 1, self.model.nvar)))
        if hasattr(self.pp_splt, 'autoscale'):
            self.pp_splt.autoscale(enable=True, axis='y', tight=True)
        self.pp_splt.legend(self.model.state_variables)
        self._plot_phase_plane()


    def _update_state_variables(self, val):
        """
        Update the default state-variable values, used for non-visualised state
        variables, based of the current slider values.
        """
        for sv in self.sv_sliders:
            k = self.model.state_variables.index(sv)
            self.default_sv[k] = self.sv_sliders[sv].val

        self._calc_phase_plane()
        self._update_phase_plane()


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
        self.refresh()


    def _set_mesh_grid(self):
        """
        Generate the phase-plane gridding based on currently selected statevariables
        and their range values.
        """
        xlo = self.model.state_variable_range[self.svx][0]
        xhi = self.model.state_variable_range[self.svx][1]
        ylo = self.model.state_variable_range[self.svy][0]
        yhi = self.model.state_variable_range[self.svy][1]

        self.X = numpy.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS * 1j)]
        self.Y = numpy.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS * 1j)]


    def _set_state_vector(self):
        """
        """
        #import pdb; pdb.set_trace()
        svr = self.model.state_variable_range
        sv_mean = numpy.array([svr[key].mean() for key in self.model.state_variables])
        sv_mean = sv_mean.reshape((self.model.nvar, 1, 1))
        self.default_sv = sv_mean.repeat(self.model.number_of_modes, axis=2)
        self.no_coupling = numpy.zeros((self.model.nvar, 1, self.model.number_of_modes))


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

        #Calculate an example trajectory
        state = self.default_sv.copy()
        state[svx_ind] = x
        state[svy_ind] = y
        scheme = self.integrator.scheme
        traj = numpy.zeros((TRAJ_STEPS + 1, self.model.nvar, 1,
                            self.model.number_of_modes))
        traj[0, :] = state
        for step in range(TRAJ_STEPS):
            #import pdb; pdb.set_trace()
            state = scheme(state, self.model.dfun, self.no_coupling, 0.0, 0.0)
            traj[step + 1, :] = state

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
