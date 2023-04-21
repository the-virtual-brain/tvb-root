# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
An interactive phase-plane plot generated from a Model object of TVB.

Optionally an Integrator object from TVB can be specified, this will be used to
generate sample trajectories -- not the phase-plane. This is mainly interesting
for visualising the effect of noise on a trajectory.

Demo::

    from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive
    ppi_fig = PhasePlaneInteractive()
    ppi_fig.show()


Example specifying a Model and stochastic sample trajectories::

    import tvb.simulator.models
    from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive
    MODEL = tvb.simulator.models.JansenRit()
    import tvb.simulator.integrators
    INTEGRATOR = tvb.simulator.integrators.HeunStochastic(dt=2**-5)
    ppi_fig = PhasePlaneInteractive(model=MODEL, integrator=INTEGRATOR)
    ppi_fig.show()


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
# TODO: Add more LOG statements.
# TODO: Add connectivity term to parameters...
# TODO: Memory grows with usage, may be just a lazy garbage collector but, should
#      check for leaks or look into "forcing" cleanup...

import numpy
import matplotlib.pyplot as plt
import colorsys
import numbers
import matplotlib.widgets as widgets
from deprecated import deprecated
from tvb.simulator.common import get_logger
import tvb.simulator.models as models_module
import tvb.simulator.integrators as integrators_module
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List

LOG = get_logger(__name__)

# Define a colour theme... see: matplotlib.colors.cnames.keys()
BACKGROUNDCOLOUR = "lightgray"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "darkred"

# Set the resolution of the phase-plane and sample trajectories.
NUMBEROFGRIDPOINTS = 42
TRAJ_STEPS = 4096


def get_color(num_colours):
    for hue in range(num_colours):
        hue = 1.0 * hue / num_colours
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)


@deprecated(reason="Use tvb-widgets instead")
class PhasePlaneInteractive(HasTraits):
    """
    The GUI for the interactive phase-plane viewer provides sliders for setting:
        - The value of all parameters of the Model.
        - The extent of the axes.
        - A fixed value for the state-variables which aren't currently selected.
        - The noise strength, if a stocahstic integrator is specified.

    and radio buttons for selecting:
        - Which state-variables to show on each axis.
        - Which mode to show, if the Model has them.

    Clicking on the phase-plane will generate a sample trajectory, originating
    from where you clicked.

    """

    model = Attr(
        field_type=models_module.Model,
        label="Model",
        default=models_module.Generic2dOscillator(),
        doc="""An instance of the local dynamic model to be investigated with
        PhasePlaneInteractive.""")

    integrator = Attr(
        field_type=integrators_module.Integrator,
        label="Integrator",
        default=integrators_module.RungeKutta4thOrderDeterministic(),
        doc="""The integration scheme used to for generating sample
        trajectories on the phase-plane. NOTE: This is not used for generating
        the phase-plane itself, ie the vector field and nulclines.""")

    exclude_sliders = List(
        of=str)

    def __init__(self, **kwargs):
        """
        Initialise based on provided keywords or their traited defaults. Also,
        initialise the place-holder attributes that aren't filled until the
        show() method is called.
        """
        super(PhasePlaneInteractive, self).__init__(**kwargs)
        LOG.debug(str(kwargs))

        # figure
        self.ipp_fig = None

        # phase-plane
        self.pp_ax = None
        self.X = None
        self.Y = None
        self.U = None
        self.V = None
        self.UVmag = None
        self.nullcline_x = None
        self.nullcline_y = None
        self.pp_quivers = None

        # Current state
        self.svx = None
        self.svy = None
        self.default_sv = None
        self.no_coupling = None
        self.mode = None

        # Selectors
        self.state_variable_x = None
        self.state_variable_y = None
        self.mode_selector = None

        # Sliders
        self.param_sliders = None
        self.axes_range_sliders = None
        self.sv_sliders = None
        self.noise_slider = None

        # Reset buttons
        self.reset_param_button = None
        self.reset_sv_button = None
        self.reset_axes_button = None
        self.reset_noise_button = None
        self.reset_seed_button = None

    def show(self):
        """ Generate the interactive phase-plane figure. """
        model_name = self.model.__class__.__name__
        msg = "Generating an interactive phase-plane plot for %s"
        LOG.info(msg % model_name)

        # Make sure the model is fully configured...
        self.model.configure()

        # Setup the inital(current) state
        self.svx = self.model.state_variables[0]  # x-axis: 1st state variable
        self.svy = self.model.state_variables[1]  # y-axis: 2nd state variable
        self.mode = 0
        self.set_state_vector()

        # Make the figure:
        self.create_figure()

        # Selectors
        self.add_state_variable_selector()
        self.add_mode_selector()

        # Sliders
        self.add_axes_range_sliders()
        self.add_state_variable_sliders()
        self.add_param_sliders()
        if isinstance(self.integrator, integrators_module.IntegratorStochastic):
            if self.integrator.noise.ntau > 0.0:
                self.integrator.noise.configure_coloured(self.integrator.dt,
                                                         (1, self.model.nvar, 1,
                                                          self.model.number_of_modes))
            else:
                self.integrator.noise.configure_white(self.integrator.dt,
                                                      (1, self.model.nvar, 1,
                                                       self.model.number_of_modes))

            self.add_noise_slider()
            self.add_reset_noise_button()
            self.add_reset_seed_button()

        # Reset buttons
        self.add_reset_param_button()
        self.add_reset_sv_button()
        self.add_reset_axes_button()

        # Calculate the phase plane
        self.set_mesh_grid()
        self.calc_phase_plane()

        # Plot phase plane
        self.plot_phase_plane()

        # add mouse handler for trajectory clicking
        self.ipp_fig.canvas.mpl_connect('button_press_event',
                                        self.click_trajectory)
        # import pdb; pdb.set_trace()

        plt.show()

    ##------------------------------------------------------------------------##
    ##----------------- Functions for building the figure --------------------##
    ##------------------------------------------------------------------------##

    def create_figure(self):
        """ Create the figure and phase-plane axes. """
        # Figure and main phase-plane axes
        model_name = self.model.__class__.__name__
        integrator_name = self.integrator.__class__.__name__
        figsize = 10, 5
        try:
            figure_window_title = "Interactive phase-plane: " + model_name
            figure_window_title += "   --   %s" % integrator_name
            self.ipp_fig = plt.figure(num=figure_window_title,
                                      figsize=figsize,
                                      facecolor=BACKGROUNDCOLOUR,
                                      edgecolor=EDGECOLOUR)
        except ValueError:
            LOG.info("My life would be easier if you'd update your PyLab...")
            self.ipp_fig = plt.figure(num=42, figsize=figsize,
                                      facecolor=BACKGROUNDCOLOUR,
                                      edgecolor=EDGECOLOUR)

        self.pp_ax = self.ipp_fig.add_axes([0.265, 0.2, 0.5, 0.75])

        self.pp_splt = self.ipp_fig.add_subplot(212)
        self.ipp_fig.subplots_adjust(left=0.265, bottom=0.02, right=0.765,
                                     top=0.3, wspace=0.1, hspace=None)
        self.pp_splt.set_prop_cycle(color=get_color(self.model.nvar))
        self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                          numpy.zeros((TRAJ_STEPS + 1, self.model.nvar)))
        if hasattr(self.pp_splt, 'autoscale'):
            self.pp_splt.autoscale(enable=True, axis='y', tight=True)
        self.pp_splt.legend(self.model.state_variables)

    def add_state_variable_selector(self):
        """
        Generate radio selector buttons to set which state variable is displayed
        on the x and y axis of the phase-plane plot.
        """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        # State variable for the x axis
        pos_shp = [0.07, 0.05, 0.065, 0.12 + 0.006 * self.model.nvar]
        rax = self.ipp_fig.add_axes(pos_shp, facecolor=AXCOLOUR, title="x-axis")
        self.state_variable_x = widgets.RadioButtons(rax,
                                                     tuple(self.model.state_variables),
                                                     active=svx_ind)
        self.state_variable_x.on_clicked(self.update_svx)

        # State variable for the y axis
        pos_shp = [0.14, 0.05, 0.065, 0.12 + 0.006 * self.model.nvar]
        rax = self.ipp_fig.add_axes(pos_shp, facecolor=AXCOLOUR, title="y-axis")
        self.state_variable_y = widgets.RadioButtons(rax,
                                                     tuple(self.model.state_variables),
                                                     active=svy_ind)
        self.state_variable_y.on_clicked(self.update_svy)

    def add_mode_selector(self):
        """
        Add a radio button to the figure for selecting which mode of the model
        should be displayed.
        """
        pos_shp = [0.02, 0.07, 0.04, 0.1 + 0.002 * self.model.number_of_modes]
        rax = self.ipp_fig.add_axes(pos_shp, facecolor=AXCOLOUR, title="Mode")
        mode_tuple = tuple(range(self.model.number_of_modes))
        self.mode_selector = widgets.RadioButtons(rax, mode_tuple, active=0)
        self.mode_selector.on_clicked(self.update_mode)

    def add_axes_range_sliders(self):
        """
        Add sliders to the figure to allow the phase-planes axes to be set.
        """
        self.axes_range_sliders = dict()

        default_range_x = (self.model.state_variable_range[self.svx][1] -
                           self.model.state_variable_range[self.svx][0])
        default_range_y = (self.model.state_variable_range[self.svy][1] -
                           self.model.state_variable_range[self.svy][0])
        min_val_x = self.model.state_variable_range[self.svx][0] - 4.0 * default_range_x
        max_val_x = self.model.state_variable_range[self.svx][1] + 4.0 * default_range_x
        min_val_y = self.model.state_variable_range[self.svy][0] - 4.0 * default_range_y
        max_val_y = self.model.state_variable_range[self.svy][1] + 4.0 * default_range_y

        sax = self.ipp_fig.add_axes([0.04, 0.835, 0.125, 0.025],
                                    facecolor=AXCOLOUR)
        sl_x_min = widgets.Slider(sax, "xlo", min_val_x, max_val_x,
                                  valinit=self.model.state_variable_range[self.svx][0])
        sl_x_min.on_changed(self.update_range)

        sax = self.ipp_fig.add_axes([0.04, 0.8, 0.125, 0.025], facecolor=AXCOLOUR)
        sl_x_max = widgets.Slider(sax, "xhi", min_val_x, max_val_x,
                                  valinit=self.model.state_variable_range[self.svx][1])
        sl_x_max.on_changed(self.update_range)

        sax = self.ipp_fig.add_axes([0.04, 0.765, 0.125, 0.025],
                                    facecolor=AXCOLOUR)
        sl_y_min = widgets.Slider(sax, "ylo", min_val_y, max_val_y,
                                  valinit=self.model.state_variable_range[self.svy][0])
        sl_y_min.on_changed(self.update_range)

        sax = self.ipp_fig.add_axes([0.04, 0.73, 0.125, 0.025], facecolor=AXCOLOUR)
        sl_y_max = widgets.Slider(sax, "yhi", min_val_y, max_val_y,
                                  valinit=self.model.state_variable_range[self.svy][1])
        sl_y_max.on_changed(self.update_range)

        self.axes_range_sliders["sl_x_min"] = sl_x_min
        self.axes_range_sliders["sl_x_max"] = sl_x_max
        self.axes_range_sliders["sl_y_min"] = sl_y_min
        self.axes_range_sliders["sl_y_max"] = sl_y_max

    def add_state_variable_sliders(self):
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
            sax = self.ipp_fig.add_axes(pos_shp, facecolor=AXCOLOUR)
            sv_str = self.model.state_variables[sv]
            self.sv_sliders[sv_str] = widgets.Slider(sax, sv_str,
                                                     msv_range[sv_str][0],
                                                     msv_range[sv_str][1],
                                                     valinit=self.default_sv[sv, 0, 0])
            self.sv_sliders[sv_str].on_changed(self.update_state_variables)

    # Traited paramaters as sliders
    def add_param_sliders(self):
        """
        Add sliders to the figure to allow the models parameters to be set.
        """
        offset = 0.0
        self.param_sliders = dict()
        # import pdb; pdb.set_trace()
        for param_name in type(self.model).declarative_attrs:
            if self.exclude_sliders is not None and param_name in self.exclude_sliders:
                continue
            param_def = getattr(type(self.model), param_name)
            if not isinstance(param_def, NArray) or not issubclass(param_def.dtype.type, numbers.Real):
                continue
            param_range = param_def.domain
            if param_range is None:
                continue
            offset += 0.035
            sax = self.ipp_fig.add_axes([0.825, 0.865 - offset, 0.125, 0.025],
                                        facecolor=AXCOLOUR)
            param_value = getattr(self.model, param_name)
            param_value = param_value[0] if len(param_value) > 0 else 0
            self.param_sliders[param_name] = widgets.Slider(sax, param_name,
                                                            param_range.lo,
                                                            param_range.hi,
                                                            valinit=param_value)
            self.param_sliders[param_name].on_changed(self.update_parameters)

    def add_noise_slider(self):
        """
        Add a slider to the figure to allow the integrators noise strength to
        be set.
        """
        pos_shp = [0.825, 0.1, 0.125, 0.025]
        sax = self.ipp_fig.add_axes(pos_shp, facecolor=AXCOLOUR)

        self.noise_slider = widgets.Slider(sax, "Log Noise", -9.0, 1.0,
                                           valinit=self.integrator.noise.nsig[0])
        self.noise_slider.on_changed(self.update_noise)

    def add_reset_param_button(self):
        """
        Add a button to the figure for reseting the model parameter values to
        their original values.
        """
        bax = self.ipp_fig.add_axes([0.825, 0.865, 0.125, 0.04])
        self.reset_param_button = widgets.Button(bax, 'Reset parameters',
                                                 color=BUTTONCOLOUR,
                                                 hovercolor=HOVERCOLOUR)

        def reset_parameters(event):
            for param_slider in self.param_sliders:
                self.param_sliders[param_slider].reset()

        self.reset_param_button.on_clicked(reset_parameters)

    def add_reset_sv_button(self):
        """
        Add a button to the figure for reseting the model state variables to
        their default values.
        """
        bax = self.ipp_fig.add_axes([0.04, 0.60, 0.125, 0.04])
        self.reset_sv_button = widgets.Button(bax, 'Reset state-variables',
                                              color=BUTTONCOLOUR,
                                              hovercolor=HOVERCOLOUR)

        def reset_state_variables(event):
            for svsl in self.sv_sliders.values():
                svsl.reset()

        self.reset_sv_button.on_clicked(reset_state_variables)

    def add_reset_noise_button(self):
        """
        Add a button to the figure for reseting the noise to its default value.
        """
        bax = self.ipp_fig.add_axes([0.825, 0.135, 0.125, 0.04])
        self.reset_noise_button = widgets.Button(bax, 'Reset noise strength',
                                                 color=BUTTONCOLOUR,
                                                 hovercolor=HOVERCOLOUR)

        def reset_noise(event):
            self.noise_slider.reset()

        self.reset_noise_button.on_clicked(reset_noise)

    def add_reset_seed_button(self):
        """
        Add a button to the figure for reseting the random number generator to
        its intial state. For reproducible noise...
        """
        bax = self.ipp_fig.add_axes([0.825, 0.05, 0.125, 0.04])
        self.reset_seed_button = widgets.Button(bax, 'Reset random stream',
                                                color=BUTTONCOLOUR,
                                                hovercolor=HOVERCOLOUR)

        def reset_seed(event):
            self.integrator.noise.trait["random_stream"].reset()

        self.reset_seed_button.on_clicked(reset_seed)

    def add_reset_axes_button(self):
        """
        Add a button to the figure for reseting the phase-plane axes to their
        default ranges.
        """
        bax = self.ipp_fig.add_axes([0.04, 0.87, 0.125, 0.04])
        self.reset_axes_button = widgets.Button(bax, 'Reset axes',
                                                color=BUTTONCOLOUR,
                                                hovercolor=HOVERCOLOUR)

        def reset_ranges(event):
            self.axes_range_sliders["sl_x_min"].reset()
            self.axes_range_sliders["sl_x_max"].reset()
            self.axes_range_sliders["sl_y_min"].reset()
            self.axes_range_sliders["sl_y_max"].reset()

        self.reset_axes_button.on_clicked(reset_ranges)

    ##------------------------------------------------------------------------##
    ##------------------- Functions for updating the figure ------------------##
    ##------------------------------------------------------------------------##

    # NOTE: All the ax.set_xlim, poly.xy, etc, garbage below is fragile. It works
    #      at the moment, but there are currently bugs in Slider and the hackery
    #      below takes these into account... If the bugs are fixed/changed then
    #      this could break. As an example, the Slider doc says poly is a
    #      Rectangle, but it's actually a Polygon. The Slider set_val method
    #      assumes a Rectangle even though this is not the case, so the array
    #      Slider.poly.xy is corrupted by that method. The corruption isn't
    #      visible in the plot, which is probably why it hasn't been fixed...

    def update_xrange_sliders(self):
        """
        A hacky update of the x-axis range sliders that is called when the
        state-variable selected for the x-axis is changed.
        """
        default_range_x = (self.model.state_variable_range[self.svx][1] -
                           self.model.state_variable_range[self.svx][0])
        min_val_x = self.model.state_variable_range[self.svx][0] - 4.0 * default_range_x
        max_val_x = self.model.state_variable_range[self.svx][1] + 4.0 * default_range_x
        self.axes_range_sliders["sl_x_min"].valinit = self.model.state_variable_range[self.svx][0]
        self.axes_range_sliders["sl_x_min"].valmin = min_val_x
        self.axes_range_sliders["sl_x_min"].valmax = max_val_x
        self.axes_range_sliders["sl_x_min"].ax.set_xlim(min_val_x, max_val_x)
        self.axes_range_sliders["sl_x_min"].poly.axes.set_xlim(min_val_x, max_val_x)
        self.axes_range_sliders["sl_x_min"].poly.xy[[0, 1], 0] = min_val_x
        self.axes_range_sliders["sl_x_min"].vline.set_data(
            ([self.axes_range_sliders["sl_x_min"].valinit, self.axes_range_sliders["sl_x_min"].valinit], [0, 1]))
        self.axes_range_sliders["sl_x_max"].valinit = self.model.state_variable_range[self.svx][1]
        self.axes_range_sliders["sl_x_max"].valmin = min_val_x
        self.axes_range_sliders["sl_x_max"].valmax = max_val_x
        self.axes_range_sliders["sl_x_max"].ax.set_xlim(min_val_x, max_val_x)
        self.axes_range_sliders["sl_x_max"].poly.axes.set_xlim(min_val_x, max_val_x)
        self.axes_range_sliders["sl_x_max"].poly.xy[[0, 1], 0] = min_val_x
        self.axes_range_sliders["sl_x_max"].vline.set_data(
            ([self.axes_range_sliders["sl_x_max"].valinit, self.axes_range_sliders["sl_x_max"].valinit], [0, 1]))
        self.axes_range_sliders["sl_x_min"].reset()
        self.axes_range_sliders["sl_x_max"].reset()

    def update_yrange_sliders(self):
        """
        A hacky update of the y-axis range sliders that is called when the
        state-variable selected for the y-axis is changed.
        """
        # svy_ind = self.model.state_variables.index(self.svy)
        default_range_y = (self.model.state_variable_range[self.svy][1] -
                           self.model.state_variable_range[self.svy][0])
        min_val_y = self.model.state_variable_range[self.svy][0] - 4.0 * default_range_y
        max_val_y = self.model.state_variable_range[self.svy][1] + 4.0 * default_range_y
        self.axes_range_sliders["sl_y_min"].valinit = self.model.state_variable_range[self.svy][0]
        self.axes_range_sliders["sl_y_min"].valmin = min_val_y
        self.axes_range_sliders["sl_y_min"].valmax = max_val_y
        self.axes_range_sliders["sl_y_min"].ax.set_xlim(min_val_y, max_val_y)
        self.axes_range_sliders["sl_y_min"].poly.axes.set_xlim(min_val_y, max_val_y)
        self.axes_range_sliders["sl_y_min"].poly.xy[[0, 1], 0] = min_val_y
        self.axes_range_sliders["sl_y_min"].vline.set_data(
            ([self.axes_range_sliders["sl_y_min"].valinit, self.axes_range_sliders["sl_y_min"].valinit], [0, 1]))
        self.axes_range_sliders["sl_y_max"].valinit = self.model.state_variable_range[self.svy][1]
        self.axes_range_sliders["sl_y_max"].valmin = min_val_y
        self.axes_range_sliders["sl_y_max"].valmax = max_val_y
        self.axes_range_sliders["sl_y_max"].ax.set_xlim(min_val_y, max_val_y)
        self.axes_range_sliders["sl_y_max"].poly.axes.set_xlim(min_val_y, max_val_y)
        self.axes_range_sliders["sl_y_max"].poly.xy[[0, 1], 0] = min_val_y
        self.axes_range_sliders["sl_y_max"].vline.set_data(
            ([self.axes_range_sliders["sl_y_max"].valinit, self.axes_range_sliders["sl_y_max"].valinit], [0, 1]))
        self.axes_range_sliders["sl_y_min"].reset()
        self.axes_range_sliders["sl_y_max"].reset()

    def update_svx(self, label):
        """
        Update state variable used for x-axis based on radio buttton selection.
        """
        self.svx = label
        self.update_xrange_sliders()
        self.set_mesh_grid()
        self.calc_phase_plane()
        self.update_phase_plane()

    def update_svy(self, label):
        """
        Update state variable used for y-axis based on radio buttton selection.
        """
        self.svy = label
        self.update_yrange_sliders()
        self.set_mesh_grid()
        self.calc_phase_plane()
        self.update_phase_plane()

    def update_mode(self, label):
        """ Update the visualised mode based on radio button selection. """
        self.mode = label
        self.update_phase_plane()

    def update_parameters(self, val):
        """
        Update model parameters based on the current parameter slider values.

        NOTE: Haven't figured out how to update independantly, so just update
            everything.
        """
        # TODO: Grab caller and use val directly, ie independent parameter update.
        # import pdb; pdb.set_trace()
        for param in self.param_sliders:
            setattr(self.model, param, numpy.array([self.param_sliders[param].val]))

        self.model.update_derived_parameters()
        self.calc_phase_plane()
        self.update_phase_plane()

    def update_noise(self, nsig):
        """ Update integrator noise based on the noise slider value. """
        self.integrator.noise.nsig = numpy.array([10 ** nsig, ])

    def update_range(self, val):
        """
        Update the axes ranges based on the current axes slider values.

        NOTE: Haven't figured out how to update independantly, so just update
            everything.

        """
        # TODO: Grab caller and use val directly, ie independent range update.
        self.axes_range_sliders["sl_x_min"].ax.set_facecolor(AXCOLOUR)
        self.axes_range_sliders["sl_x_max"].ax.set_facecolor(AXCOLOUR)
        self.axes_range_sliders["sl_y_min"].ax.set_facecolor(AXCOLOUR)
        self.axes_range_sliders["sl_y_max"].ax.set_facecolor(AXCOLOUR)

        if (self.axes_range_sliders["sl_x_min"].val >=
                self.axes_range_sliders["sl_x_max"].val):
            LOG.error("X-axis min must be less than max...")
            self.axes_range_sliders["sl_x_min"].ax.set_facecolor("Red")
            self.axes_range_sliders["sl_x_max"].ax.set_facecolor("Red")
            return
        if (self.axes_range_sliders["sl_y_min"].val >=
                self.axes_range_sliders["sl_y_max"].val):
            LOG.error("Y-axis min must be less than max...")
            self.axes_range_sliders["sl_y_min"].ax.set_facecolor("Red")
            self.axes_range_sliders["sl_y_max"].ax.set_facecolor("Red")
            return

        msv_range = self.model.state_variable_range
        msv_range[self.svx][0] = self.axes_range_sliders["sl_x_min"].val
        msv_range[self.svx][1] = self.axes_range_sliders["sl_x_max"].val
        msv_range[self.svy][0] = self.axes_range_sliders["sl_y_min"].val
        msv_range[self.svy][1] = self.axes_range_sliders["sl_y_max"].val
        self.set_mesh_grid()
        self.calc_phase_plane()
        self.update_phase_plane()

    def update_phase_plane(self):
        """ Clear the axes and redraw the phase-plane. """
        self.pp_ax.clear()
        self.pp_splt.clear()
        self.pp_splt.set_prop_cycle('color', get_color(self.model.nvar))
        self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                          numpy.zeros((TRAJ_STEPS + 1, self.model.nvar)))
        if hasattr(self.pp_splt, 'autoscale'):
            self.pp_splt.autoscale(enable=True, axis='y', tight=True)
        self.pp_splt.legend(self.model.state_variables)
        self.plot_phase_plane()

    def update_state_variables(self, val):
        """
        Update the default state-variable values, used for non-visualised state
        variables, based of the current slider values.
        """
        for sv in self.sv_sliders:
            k = self.model.state_variables.index(sv)
            self.default_sv[k] = self.sv_sliders[sv].val

        self.calc_phase_plane()
        self.update_phase_plane()

    def set_mesh_grid(self):
        """
        Generate the phase-plane gridding based on currently selected
        state-variables and their range values.
        """
        xlo = self.model.state_variable_range[self.svx][0]
        xhi = self.model.state_variable_range[self.svx][1]
        ylo = self.model.state_variable_range[self.svy][0]
        yhi = self.model.state_variable_range[self.svy][1]

        self.X = numpy.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS * 1j)]
        self.Y = numpy.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS * 1j)]

    def set_state_vector(self):
        """
        Set up a vector containing the default state-variable values and create
        a filler(all zeros) for the coupling arg of the Model's dfun method.
        This method is called once at initialisation (show()).
        """
        # import pdb; pdb.set_trace()
        sv_mean = numpy.array([self.model.state_variable_range[key].mean() for key in self.model.state_variables])
        sv_mean = sv_mean.reshape((self.model.nvar, 1, 1))
        self.default_sv = sv_mean.repeat(self.model.number_of_modes, axis=2)
        self.no_coupling = numpy.zeros((self.model.nvar, 1,
                                        self.model.number_of_modes))

    def calc_phase_plane(self):
        """ Calculate the vector field. """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        # Calculate the vector field discretely sampled at a grid of points
        grid_point = self.default_sv.copy()
        self.U = numpy.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS,
                              self.model.number_of_modes))
        self.V = numpy.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS,
                              self.model.number_of_modes))
        for ii in range(NUMBEROFGRIDPOINTS):
            grid_point[svy_ind] = self.Y[ii]
            for jj in range(NUMBEROFGRIDPOINTS):
                # import pdb; pdb.set_trace()
                grid_point[svx_ind] = self.X[jj]

                d = self.model.dfun(grid_point, self.no_coupling)

                for kk in range(self.model.number_of_modes):
                    self.U[ii, jj, kk] = d[svx_ind, 0, kk]
                    self.V[ii, jj, kk] = d[svy_ind, 0, kk]

        # Colours for the vector field quivers
        # self.UVmag = numpy.sqrt(self.U**2 + self.V**2)

        # import pdb; pdb.set_trace()
        if numpy.isnan(self.U).any() or numpy.isnan(self.V).any():
            LOG.error("NaN")

    def plot_phase_plane(self):
        """ Plot the vector field and its nullclines. """
        # Set title and axis labels
        model_name = self.model.__class__.__name__
        self.pp_ax.set(title=model_name + " mode " + str(self.mode))
        self.pp_ax.set(xlabel="State Variable " + self.svx)
        self.pp_ax.set(ylabel="State Variable " + self.svy)

        # import pdb; pdb.set_trace()
        # Plot a discrete representation of the vector field
        if numpy.all(self.U[:, :, self.mode] + self.V[:, :, self.mode] == 0):
            self.pp_ax.set(title=model_name + " mode " + str(self.mode) + ": NO MOTION IN THIS PLANE")
            X, Y = numpy.meshgrid(self.X, self.Y)
            self.pp_quivers = self.pp_ax.scatter(X, Y, s=8, marker=".", c="k")
        else:
            self.pp_quivers = self.pp_ax.quiver(self.X, self.Y,
                                                self.U[:, :, self.mode],
                                                self.V[:, :, self.mode],
                                                # self.UVmag[:, :, self.mode],
                                                width=0.001, headwidth=8)

        # Plot the nullclines
        self.nullcline_x = self.pp_ax.contour(self.X, self.Y,
                                              self.U[:, :, self.mode],
                                              [0], colors="r")
        self.nullcline_y = self.pp_ax.contour(self.X, self.Y,
                                              self.V[:, :, self.mode],
                                              [0], colors="g")
        plt.draw()

    def plot_trajectory(self, x, y):
        """
        Plot a sample trajectory, starting at the position x,y in the
        phase-plane. This method is called as a result of a mouse click on the
        phase-plane.
        """
        svx_ind = self.model.state_variables.index(self.svx)
        svy_ind = self.model.state_variables.index(self.svy)

        # Calculate an example trajectory
        state = self.default_sv.copy()
        self.integrator.clamped_state_variable_indices = numpy.setdiff1d(
            numpy.r_[:len(self.model.state_variables)], numpy.r_[svx_ind, svy_ind])
        self.integrator.clamped_state_variable_values = self.default_sv[self.integrator.clamped_state_variable_indices]
        state[svx_ind] = x
        state[svy_ind] = y
        scheme = self.integrator.scheme
        traj = numpy.zeros((TRAJ_STEPS + 1, self.model.nvar, 1,
                            self.model.number_of_modes))
        traj[0, :] = state
        for step in range(TRAJ_STEPS):
            # import pdb; pdb.set_trace()
            state = scheme(state, self.model.dfun, self.no_coupling, 0.0, 0.0)
            traj[step + 1, :] = state

        self.pp_ax.scatter(x, y, s=42, c='g', marker='o', edgecolor=None)
        self.pp_ax.plot(traj[:, svx_ind, 0, self.mode],
                        traj[:, svy_ind, 0, self.mode])

        # Plot the selected state variable trajectories as a function of time
        self.pp_splt.plot(numpy.arange(TRAJ_STEPS + 1) * self.integrator.dt,
                          traj[:, :, 0, self.mode])

        plt.draw()

    def click_trajectory(self, event):
        """
        This method captures mouse clicks on the phase-plane and then uses the
        plot_trajectory() method to generate a sample trajectory.
        """
        if event.inaxes is self.pp_ax:
            x, y = event.xdata, event.ydata
            LOG.info('trajectory starting at (%f, %f)', x, y)
            self.plot_trajectory(x, y)


def _list_of_models():
    base = models_module.Model
    for key in dir(models_module):
        attr = getattr(models_module, key)
        if isinstance(attr, type) and issubclass(attr, base):
            if attr is base:
                continue
            first_para = attr.__doc__.replace('\n', ' ').replace('\t', ' ')[:100] + ' ...'
            yield attr.__name__, attr._ui_name


if __name__ == "__main__":

    import sys

    try:
        Model = getattr(models_module, sys.argv[1])
    except Exception:
        print("""
usage: python -m tvb.simulator.plot.phase_plane_interactive name_of_model

where name_of_model is one of

%s
        """ % (
            '\n'.join(map('{0[0]:>25} - {0[1]}'.format, _list_of_models()))
        ))
        sys.exit(1)

    ppi_fig = PhasePlaneInteractive(model=Model())
    ppi_fig.show()
