# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
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
An interactive phase-plane plot generated from a Model object of TVB.

Optionally an Integrator object from TVB can be specified, this will be used to
generate sample trajectories -- not the phase-plane. This is mainly interesting
for visualising the effect of noise on a trajectory.

Demo::

    import tvb.simulator.plot.phase_plane_interactive as ppi
    ppi.plot_pp()
"""
from ipywidgets.widgets.interaction import interactive_output
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator.lab import integrators
import colorsys
import tvb.simulator.models as models_module
import tvb.simulator.integrators as integrators_module

def get_color(num_colours):
    for hue in range(num_colours):
        hue = 1.0 * hue / num_colours
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)

class PhasePlaneInteractive:
    NUMBEROFGRIDPOINTS = 42
    TRAJ_STEPS = 4096
    exclude_sliders = None

    def __init__(self, model = models_module.Generic2dOscillator(), integrator = integrators_module.RungeKutta4thOrderDeterministic()):
        self.model = model
        self.integrator = integrator

        self.params = dict()

        self.svx = model.state_variables[0] #x-axis: 1st state variable
        self.svy = model.state_variables[1] #y-axis: 2nd state variable
        self.mode = 0

        # LAYOUTS
        self.slider_layout = widgets.Layout(width='90%')
        self.slider_style = {'description_width': 'initial'}
        self.button_layout = widgets.Layout(width='90%')
        self.other_layout = widgets.Layout(width='90%')
        self.box_layout = widgets.Layout(border='solid 1px black',
                                        margin='0px 5px 5px 0px',
                                        padding='2px 2px 2px 2px')
        
        self.traj_out = None
    
    def create_ui(self):
        #Figure and main phase-plane axes
        self.set_state_vector()

        self.add_axes_sliders()
        self.add_reset_axes_button()
        
        self.add_param_sliders()
        self.add_reset_param_button()
        
        self.add_sv_sliders()
        self.add_reset_sv_button()
        
        self.add_sv_selector()
        self.add_mode_selector()

        self.add_traj_coords_text()
        # items in first vbox
        self.mode_selector_widget = widgets.VBox([widgets.Label('Mode Selector'), self.mode_selector])
        self.svx_widget = widgets.VBox([widgets.Label('SVX Selector'), self.state_variable_x])
        self.svy_widget = widgets.VBox([widgets.Label('SVY Selector'), self.state_variable_y])
        self.ax_widgets_list = [self.reset_axes_button, self.sl_x_min, self.sl_x_max, self.sl_y_min, self.sl_y_max, 
                                    self.mode_selector_widget, self.svx_widget, self.svy_widget]

        if isinstance(self.integrator, integrators.IntegratorStochastic):
            if self.integrator.noise.ntau > 0.0:
                self.integrator.noise.configure_coloured(self.integrator.dt,
                                                         (1, self.model.nvar, 1,
                                                          self.model.number_of_modes))
            else:
                self.integrator.noise.configure_white(self.integrator.dt,
                                                      (1, self.model.nvar, 1,
                                                       self.model.number_of_modes))

            # RESET NOISE BUTTON
            self.reset_noise_button = widgets.Button(description='Reset noise strength',
                                                    disabled=False, 
                                                    layout=self.button_layout)
            def reset_noise(sad):
                self.noise_slider.value = self.noise_slider_valinit

            self.reset_noise_button.on_click(reset_noise)

            # NOISE SLIDER
            self.noise_slider_valinit = self.integrator.noise.nsig
            self.noise_slider = widgets.FloatSlider(description="Log Noise", 
                                                    min=-9.0,
                                                    max=1.0,
                                                    value = self.integrator.noise.nsig,
                                                    layout=self.slider_layout,
                                                    style = self.slider_style)

            # RESET SEED BUTTON
            self.reset_seed_button = widgets.Button(description='Reset random stream',
                                                    disabled=False,
                                                    layout=self.button_layout)
            def reset_seed(event):
                self.integrator.noise.reset_random_stream()

            self.reset_seed_button.on_click(reset_seed)

            self.ax_widgets_list.extend([self.reset_noise_button, self.noise_slider, self.reset_seed_button])

        self.ax_widgets = widgets.VBox(self.ax_widgets_list, layout=self.box_layout)
        self.sv_widgets = widgets.VBox([self.reset_sv_button]+list(self.sv_sliders.values())+[self.traj_label, self.traj_text, self.traj_out], layout=self.box_layout)
        self.param_widgets = widgets.VBox([self.reset_param_button]+list(self.param_sliders.values()), layout=self.box_layout)

        items = [self.param_widgets, self.sv_widgets, self.ax_widgets]
        grid = widgets.GridBox(items, layout=widgets.Layout(grid_template_columns="326px 326px 326px"))
        return grid

    def set_state_vector(self):
        # SET STATE VECTOR
        self.sv_mean = np.array([self.model.state_variable_range[key].mean() for key in self.model.state_variables])
        self.sv_mean = self.sv_mean.reshape((self.model.nvar, 1, 1))
        self.default_sv = self.sv_mean.repeat(self.model.number_of_modes, axis=2)
        self.no_coupling = np.zeros((self.model.nvar, 1, self.model.number_of_modes))
    
    def add_reset_axes_button(self):
        
        def reset_ranges(val):
            self.sl_x_min.value = self.sl_x_min_initval
            self.sl_x_max.value = self.sl_x_max_initval
            self.sl_y_min.value = self.sl_y_min_initval
            self.sl_y_max.value = self.sl_y_max_initval
    
        self.reset_axes_button = widgets.Button(description='Reset axes',
                                                disabled=False, 
                                                layout=self.button_layout)
        self.reset_axes_button.on_click(reset_ranges)

    def add_axes_sliders(self):
        default_range_x = (self.model.state_variable_range[self.svx][1] -
                            self.model.state_variable_range[self.svx][0])
        default_range_y = (self.model.state_variable_range[self.svy][1] -
                            self.model.state_variable_range[self.svy][0])
        min_val_x = self.model.state_variable_range[self.svx][0] - 4.0 * default_range_x
        max_val_x = self.model.state_variable_range[self.svx][1] + 4.0 * default_range_x
        min_val_y = self.model.state_variable_range[self.svy][0] - 4.0 * default_range_y
        max_val_y = self.model.state_variable_range[self.svy][1] + 4.0 * default_range_y
        
        self.sl_x_min_initval = self.model.state_variable_range[self.svx][0]
        self.sl_x_max_initval = self.model.state_variable_range[self.svx][1]
        self.sl_y_min_initval = self.model.state_variable_range[self.svy][0]
        self.sl_y_max_initval = self.model.state_variable_range[self.svy][1]

        self.sl_x_min = widgets.FloatSlider(description="xlo",
                                    min=min_val_x,
                                    max=max_val_x,
                                    value= self.model.state_variable_range[self.svx][0],
                                    layout=self.slider_layout,
                                    style = self.slider_style,
                                    continuous_update=False)
        self.sl_x_min.observe(self.update_sl_x_range, 'value')

        self.sl_x_max = widgets.FloatSlider(description="xhi",
                                        min=min_val_x,
                                        max=max_val_x,
                                        value=self.model.state_variable_range[self.svx][1],
                                        layout= self.slider_layout,
                                        style = self.slider_style,
                                        continuous_update=False)

        self.sl_y_min = widgets.FloatSlider(description="ylo", 
                                        min=min_val_y, 
                                        max=max_val_y, 
                                        value=self.model.state_variable_range[self.svy][0],
                                        layout=self.slider_layout,
                                        style = self.slider_style,
                                        continuous_update=False)
        self.sl_y_min.observe(self.update_sl_y_range, 'value')

        self.sl_y_max = widgets.FloatSlider(description="yhi", 
                                        min=min_val_y, 
                                        max=max_val_y,
                                        value=self.model.state_variable_range[self.svy][1],
                                        layout=self.slider_layout,
                                        style = self.slider_style,
                                        continuous_update=False)

        self.params['sl_x_min'] = self.sl_x_min
        self.params['sl_x_max'] = self.sl_x_max
        self.params['sl_y_min'] = self.sl_y_min
        self.params['sl_y_max'] = self.sl_y_max

    def add_reset_sv_button(self):
        self.reset_sv_button = widgets.Button(description='Reset state-variables',
                                        disabled=False, 
                                        layout=self.button_layout)
            
        def reset_state_variables(sad):
            for sv in range(self.model.nvar):
                sv_str = self.model.state_variables[sv]
                self.sv_sliders[sv_str].value = self.sv_sliders_values[sv_str]

        self.reset_sv_button.on_click(reset_state_variables)

    def add_sv_sliders(self):
        msv_range = self.model.state_variable_range
        self.sv_sliders = dict()
        self.sv_sliders_values = dict()
        for sv in range(self.model.nvar):
            sv_str = self.model.state_variables[sv]
            self.sv_sliders[sv_str] = widgets.FloatSlider(description=sv_str,
                                                            min=msv_range[sv_str][0],
                                                            max=msv_range[sv_str][1],
                                                            value = self.default_sv[sv,0,0],
                                                            layout=self.slider_layout,
                                                            style = self.slider_style,
                                                            continuous_update=False)
            self.sv_sliders_values[sv_str] = self.default_sv[sv,0,0]
            self.params[sv_str] = self.sv_sliders[sv_str]

    def add_mode_selector(self):
        self.mode_tuple = tuple(range(self.model.number_of_modes))
        self.mode_selector = widgets.Dropdown(options=self.mode_tuple, value=0, layout=self.other_layout)
        self.params['mode'] = self.mode_selector

    def add_sv_selector(self):
        self.state_variable_x = widgets.Dropdown(options=list(self.model.state_variables), value=self.svx, layout=self.other_layout)
        self.params['svx'] = self.state_variable_x
        self.state_variable_x.observe(self.update_axis_sliders, 'value')

        #State variable for the y axis
        self.state_variable_y = widgets.Dropdown(options=list(self.model.state_variables), value=self.svy, layout=self.other_layout)
        self.state_variable_y.observe(self.update_axis_sliders, 'value')
        self.params['svy'] = self.state_variable_y

    def add_reset_param_button(self):
        self.reset_param_button = widgets.Button(description='Reset parameters',
                                            disabled=False,
                                            layout=self.button_layout)

        def reset_parameters(sad):
            for param_slider in self.param_sliders:
                self.param_sliders[param_slider].value = self.param_sliders_values[param_slider]

        self.reset_param_button.on_click(reset_parameters)

    def add_param_sliders(self):
        self.param_sliders = dict()
        self.param_sliders_values = dict()

        for param_name in type(self.model).declarative_attrs:
            if self.exclude_sliders is not None and param_name in self.exclude_sliders:
                continue
            param_def = getattr(type(self.model), param_name)
            if not isinstance(param_def, NArray) or not param_def.dtype == np.float :
                continue
            param_range = param_def.domain
            if param_range is None:
                continue
            param_value = getattr(self.model, param_name)[0]
            self.param_sliders[param_name] = widgets.FloatSlider(description=param_name,
                                                                min=param_range.lo,
                                                                max=param_range.hi,
                                                                value=param_value,
                                                                layout=self.slider_layout,
                                                                style = self.slider_style,
                                                                continuous_update=False)
            self.param_sliders_values[param_name] = param_value
            self.params[param_name] = self.param_sliders[param_name]
    
    def add_traj_coords_text(self):
        self.traj_label = widgets.Label('Trajectory Co-ordinates (Float)')
        self.traj_text = widgets.Text(placeholder='x,y', continuous_update=False)

        def update_traj_text(val):
            if len(self.traj_text.value) >= 3 and ',' in self.traj_text.value:
                self.traj_out.value = f'{self.traj_out.value}\nTrajectory plotted at ({self.traj_text.value}).'

        self.traj_out = widgets.Textarea(value='', placeholder='Trajectory Co-ordinates output will be shown here')
        self.params['traj_text'] = self.traj_text
        self.traj_text.observe(update_traj_text, 'value')

    def update_sl_x_range(self, val):
        self.sl_x_min.max = self.sl_x_max.value
    
    def update_sl_y_range(self, val):
        self.sl_y_min.max = self.sl_y_max.value

    def set_default_axes_sliders(self):
        default_range_x = (self.model.state_variable_range[self.state_variable_x.value][1] -
                            self.model.state_variable_range[self.state_variable_x.value][0])
        default_range_y = (self.model.state_variable_range[self.state_variable_y.value][1] -
                            self.model.state_variable_range[self.state_variable_y.value][0])
        min_val_x = self.model.state_variable_range[self.state_variable_x.value][0] - 4.0 * default_range_x
        max_val_x = self.model.state_variable_range[self.state_variable_x.value][1] + 4.0 * default_range_x
        min_val_y = self.model.state_variable_range[self.state_variable_y.value][0] - 4.0 * default_range_y
        max_val_y = self.model.state_variable_range[self.state_variable_y.value][1] + 4.0 * default_range_y

        return min_val_x,max_val_x,min_val_y,max_val_y

    def update_axis_sliders(self, val):
        
        self.sl_x_min_initval = self.model.state_variable_range[self.state_variable_x.value][0]
        self.sl_x_max_initval = self.model.state_variable_range[self.state_variable_x.value][1]
        self.sl_y_min_initval = self.model.state_variable_range[self.state_variable_y.value][0]
        self.sl_y_max_initval = self.model.state_variable_range[self.state_variable_y.value][1]

        min_val_x, max_val_x, min_val_y, max_val_y = self.set_default_axes_sliders()
        
        self.sl_x_min.min = min_val_x
        self.sl_x_min.value = self.sl_x_min_initval
        self.sl_x_min.max = max_val_x
        self.sl_x_max.min = min_val_x
        self.sl_x_max.value = self.sl_x_max_initval
        self.sl_x_max.max = max_val_x

        self.sl_y_min.min = min_val_y
        self.sl_y_min.value = self.sl_y_min_initval
        self.sl_y_min.max = max_val_y
        self.sl_y_max.min = min_val_y
        self.sl_y_max.value = self.sl_y_max_initval
        self.sl_y_max.max = max_val_y

    def get_coords(self):
        coords = self.traj_text.value.split(',')
        x_coord = float(coords[0])
        y_coord = float(coords[1])
        return x_coord, y_coord

    def show(self):
        model = self.model
        integrator = self.integrator
        TRAJ_STEPS = self.TRAJ_STEPS
        NUMBEROFGRIDPOINTS = self.NUMBEROFGRIDPOINTS
        traj_texts = []

        def printer(**plot_params):

            model_name = model.__class__.__name__
            integrator_name = integrator.__class__.__name__
            figsize = 4, 5
            try:
                figure_window_title = "Interactive phase-plane: " + model_name
                figure_window_title += "   --   %s" % integrator_name
                ipp_fig = plt.figure(num = figure_window_title,
                                            figsize = figsize)
            except ValueError:
                ipp_fig = plt.figure(num = 42, figsize = figsize)

            pp_ax = ipp_fig.add_axes([0.15, 0.2, 0.8, 0.75])
            pp_splt = ipp_fig.add_subplot(212)
            ipp_fig.subplots_adjust(left=0.15, bottom=0.02, right=0.95,
                                        top=0.3, wspace=0.1, hspace=None)
            pp_splt.set_prop_cycle(color=get_color(model.nvar))
            pp_splt.plot(np.arange(TRAJ_STEPS+1) * integrator.dt,
                            np.zeros((TRAJ_STEPS+1, model.nvar)))
            if hasattr(pp_splt, 'autoscale'):
                pp_splt.autoscale(enable=True, axis='y', tight=True)
            pp_splt.legend(model.state_variables)

            svx = plot_params.pop('svx')
            svy = plot_params.pop('svy')
            
            mode = plot_params.pop('mode')

            traj_text = plot_params.pop('traj_text')
            if len(traj_text) >=3 and ',' in traj_text:
                traj_texts.append(traj_text)

            # set model params
            for k, v in plot_params.items():
                setattr(model, k, np.r_[v])

            # state vector
            sv_mean = np.array([plot_params[key] for key in model.state_variables])
            sv_mean = sv_mean.reshape((model.nvar, 1, 1))
            default_sv = sv_mean.repeat(model.number_of_modes, axis=2)
            no_coupling = np.zeros((model.nvar, 1, model.number_of_modes))

            # Set Mesh Grid
            xlo = plot_params['sl_x_min']#model.state_variable_range[svx][0]
            xhi = plot_params['sl_x_max']#model.state_variable_range[svx][1]
            ylo = plot_params['sl_y_min']#model.state_variable_range[svy][0]
            yhi = plot_params['sl_y_max']#model.state_variable_range[svy][1]

            X = np.mgrid[xlo:xhi:(NUMBEROFGRIDPOINTS*1j)]
            Y = np.mgrid[ylo:yhi:(NUMBEROFGRIDPOINTS*1j)]
            
            # Calculate Phase Plane
            svx_ind = model.state_variables.index(svx)
            svy_ind = model.state_variables.index(svy)

            # Calculate the vector field discretely sampled at a grid of points
            grid_point = default_sv.copy()
            U = np.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS,
                                    model.number_of_modes))
            V = np.zeros((NUMBEROFGRIDPOINTS, NUMBEROFGRIDPOINTS,
                                    model.number_of_modes))
            for ii in range(NUMBEROFGRIDPOINTS):
                grid_point[svy_ind] = Y[ii]
                for jj in range(NUMBEROFGRIDPOINTS):
                    # import pdb; pdb.set_trace()
                    grid_point[svx_ind] = X[jj]

                    d = model.dfun(grid_point, no_coupling)

                    for kk in range(model.number_of_modes):
                        U[ii, jj, kk] = d[svx_ind, 0, kk]
                        V[ii, jj, kk] = d[svy_ind, 0, kk]
            
            model_name = model.__class__.__name__
            pp_ax.set(title = model_name + " mode " + str(mode))
            pp_ax.set(xlabel = "State Variable " + svx)
            pp_ax.set(ylabel = "State Variable " + svy)

            #import pdb; pdb.set_trace()
            #Plot a discrete representation of the vector field
            if np.all(U[:, :, mode] + V[:, :, mode]  == 0):
                pp_ax.set(title = model_name + " mode " + str(mode) + ": NO MOTION IN THIS PLANE")
                X, Y = np.meshgrid(X, Y)
                pp_quivers = pp_ax.scatter(X, Y, s=8, marker=".", c="k")
            else:
                pp_quivers = pp_ax.quiver(X, Y,
                                                    U[:, :, mode],
                                                    V[:, :, mode],
                                                    #UVmag[:, :, mode],
                                                    width=0.001, headwidth=8)

            #Plot the nullclines
            nullcline_x = pp_ax.contour(X, Y,
                                                U[:, :, mode],
                                                [0], colors="r")
            nullcline_y = pp_ax.contour(X, Y,
                                                V[:, :, mode],
                                                [0], colors="g")

            if len(traj_texts):
                for traj_text in traj_texts:
                    coords = traj_text.split(',')

                    x = float(coords[0])
                    y = float(coords[1])
                    svx_ind = model.state_variables.index(svx)
                    svy_ind = model.state_variables.index(svy)

                    #Calculate an example trajectory
                    state = default_sv.copy()
                    integrator.clamped_state_variable_indices = np.setdiff1d(
                        np.r_[:len(model.state_variables)], np.r_[svx_ind, svy_ind])
                    integrator.clamped_state_variable_values = default_sv[integrator.clamped_state_variable_indices]
                    state[svx_ind] = x
                    state[svy_ind] = y
                    scheme = integrator.scheme
                    traj = np.zeros((TRAJ_STEPS+1, model.nvar, 1,
                                        model.number_of_modes))
                    traj[0, :] = state
                    for step in range(TRAJ_STEPS):
                        #import pdb; pdb.set_trace()
                        state = scheme(state, model.dfun, no_coupling, 0.0, 0.0)
                        traj[step+1, :] = state

                    pp_ax.scatter(x, y, s=42, c='g', marker='o', edgecolor=None)
                    pp_ax.plot(traj[:, svx_ind, 0, mode],
                                    traj[:, svy_ind, 0, mode])

                    #Plot the selected state variable trajectories as a function of time
                    pp_splt.plot(np.arange(TRAJ_STEPS+1) * integrator.dt,
                                    traj[:, :, 0, mode])

        ui = self.create_ui()
        out = interactive_output(printer, self.params)
        display(ui, out)