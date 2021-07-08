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

def plot_pp(model = models_module.Generic2dOscillator(), integrator = integrators_module.RungeKutta4thOrderDeterministic()):

    params = {}

    # CURRENT STATE
    model = model
    integrator = integrator
    NUMBEROFGRIDPOINTS = 42
    TRAJ_STEPS = 4096
    exclude_sliders = None

    svx = model.state_variables[0] #x-axis: 1st state variable
    svy = model.state_variables[1] #y-axis: 2nd state variable
    mode = 0

    # SET STATE VECTOR
    sv_mean = np.array([model.state_variable_range[key].mean() for key in model.state_variables])
    sv_mean = sv_mean.reshape((model.nvar, 1, 1))
    default_sv = sv_mean.repeat(model.number_of_modes, axis=2)
    no_coupling = np.zeros((model.nvar, 1, model.number_of_modes))

    # LAYOUTS
    slider_layout = widgets.Layout(width='90%')
    slider_style = {'description_width': 'initial'}
    button_layout = widgets.Layout(width='90%')
    other_layout = widgets.Layout(width='90%')
    box_layout = widgets.Layout(border='solid 1px black',
                            margin='0px 5px 5px 0px',
                            padding='2px 2px 2px 2px')

    ## WIDGETS

    # RESET AXES BUTTON
    def reset_ranges(sad):
        sl_x_min.value = sl_x_min_initval
        sl_x_max.value = sl_x_max_initval
        sl_y_min.value = sl_y_min_initval
        sl_y_max.value = sl_y_max_initval

    reset_axes_button = widgets.Button(description='Reset axes',
                                            disabled=False, 
                                            layout=button_layout)
    reset_axes_button.on_click(reset_ranges)

    # AXES SLIDERS
    def update_axes_sliders(svx,svy):
        default_range_x = (model.state_variable_range[svx][1] -
                            model.state_variable_range[svx][0])
        default_range_y = (model.state_variable_range[svy][1] -
                            model.state_variable_range[svy][0])
        min_val_x = model.state_variable_range[svx][0] - 4.0 * default_range_x
        max_val_x = model.state_variable_range[svx][1] + 4.0 * default_range_x
        min_val_y = model.state_variable_range[svy][0] - 4.0 * default_range_y
        max_val_y = model.state_variable_range[svy][1] + 4.0 * default_range_y
        
        return min_val_x,max_val_x,min_val_y,max_val_y
    
    min_val_x, max_val_x, min_val_y, max_val_y = update_axes_sliders(svx,svy)

    sl_x_min_initval = model.state_variable_range[svx][0]
    sl_x_max_initval = model.state_variable_range[svx][1]
    sl_y_min_initval = model.state_variable_range[svy][0]
    sl_y_max_initval = model.state_variable_range[svy][1]

    def update_sl_x_range(val):
        sl_x_min.max = sl_x_max.value
    
    def update_sl_y_range(val):
        sl_y_min.max = sl_y_max.value

    sl_x_min = widgets.FloatSlider(description="xlo",
                                    min=min_val_x,
                                    max=max_val_x,
                                    value= model.state_variable_range[svx][0],
                                    layout=slider_layout,
                                    style = slider_style,
                                    continuous_update=False)
    sl_x_min.observe(update_sl_x_range, 'value')

    sl_x_max = widgets.FloatSlider(description="xhi",
                                    min=min_val_x,
                                    max=max_val_x,
                                    value=model.state_variable_range[svx][1],
                                    layout=slider_layout,
                                    style = slider_style,
                                    continuous_update=False)

    sl_y_min = widgets.FloatSlider(description="ylo", 
                                    min=min_val_y, 
                                    max=max_val_y, 
                                    value=model.state_variable_range[svy][0],
                                    layout=slider_layout,
                                    style = slider_style,
                                    continuous_update=False)
    sl_y_min.observe(update_sl_y_range, 'value')

    sl_y_max = widgets.FloatSlider(description="yhi", 
                                    min=min_val_y, 
                                    max=max_val_y,
                                    value=model.state_variable_range[svy][1],
                                    layout=slider_layout,
                                    style = slider_style,
                                    continuous_update=False)

    params['sl_x_min'] = sl_x_min
    params['sl_x_max'] = sl_x_max
    params['sl_y_min'] = sl_y_min
    params['sl_y_max'] = sl_y_max

    # RESET SV BUTTON
    reset_sv_button = widgets.Button(description='Reset state-variables',
                                      disabled=False, 
                                      layout=button_layout)
        
    def reset_state_variables(sad):
        for sv in range(model.nvar):
            sv_str = model.state_variables[sv]
            sv_sliders[sv_str].value = sv_sliders_values[sv_str]

    reset_sv_button.on_click(reset_state_variables)

    # SV SLIDERS
    msv_range = model.state_variable_range
    sv_sliders = dict()
    sv_sliders_values = dict()
    for sv in range(model.nvar):
        sv_str = model.state_variables[sv]
        sv_sliders[sv_str] = widgets.FloatSlider(description=sv_str,
                                                        min=msv_range[sv_str][0],
                                                        max=msv_range[sv_str][1],
                                                        value = default_sv[sv,0,0],
                                                        layout=slider_layout,
                                                        style = slider_style,
                                                        continuous_update=False)
        sv_sliders_values[sv_str] = default_sv[sv,0,0]
        params[sv_str] = sv_sliders[sv_str]

    # MODE BUTTON
    mode_tuple = tuple(range(model.number_of_modes))
    mode_selector = widgets.Dropdown(options=mode_tuple, value=0, layout=other_layout)
    params['mode'] = mode_selector

    def update_axis_sliders(val):
        nonlocal sl_x_min_initval
        nonlocal sl_x_max_initval
        nonlocal sl_y_min_initval
        nonlocal sl_y_max_initval
        nonlocal sl_x_min
        nonlocal sl_x_max
        nonlocal sl_y_min
        nonlocal sl_y_max
        
        sl_x_min_initval = model.state_variable_range[state_variable_x.value][0]
        sl_x_max_initval = model.state_variable_range[state_variable_x.value][1]
        sl_y_min_initval = model.state_variable_range[state_variable_y.value][0]
        sl_y_max_initval = model.state_variable_range[state_variable_y.value][1]

        min_val_x, max_val_x, min_val_y, max_val_y = update_axes_sliders(state_variable_x.value,state_variable_y.value)
        
        sl_x_min.min = min_val_x
        sl_x_min.value = sl_x_min_initval
        sl_x_min.max = max_val_x
        sl_x_max.min = min_val_x
        sl_x_max.value = sl_x_max_initval
        sl_x_max.max = max_val_x

        sl_y_min.min = min_val_y
        sl_y_min.value = sl_y_min_initval
        sl_y_min.max = max_val_y
        sl_y_max.min = min_val_y
        sl_y_max.value = sl_y_max_initval
        sl_y_max.max = max_val_y
        

    # SV BUTTONS
    #State variable for the x axis
    state_variable_x = widgets.Dropdown(options=list(model.state_variables), value=svx, layout=other_layout)
    params['svx'] = state_variable_x
    state_variable_x.observe(update_axis_sliders, 'value')

    #State variable for the y axis
    state_variable_y = widgets.Dropdown(options=list(model.state_variables), value=svy, layout=other_layout)
    state_variable_y.observe(update_axis_sliders, 'value')
    params['svy'] = state_variable_y

    # RESET PARAM BUTTON
    reset_param_button = widgets.Button(description='Reset parameters',
                                            disabled=False,
                                            layout=button_layout)

    def reset_parameters(sad):
        for param_slider in param_sliders:
            param_sliders[param_slider].value = param_sliders_values[param_slider]

    reset_param_button.on_click(reset_parameters)

    # PARAM SLIDER
    param_sliders = dict()
    param_sliders_values = dict()

    for param_name in type(model).declarative_attrs:
        if exclude_sliders is not None and param_name in exclude_sliders:
            continue
        param_def = getattr(type(model), param_name)
        if not isinstance(param_def, NArray) or not param_def.dtype == np.float :
            continue
        param_range = param_def.domain
        if param_range is None:
            continue
        param_value = getattr(model, param_name)[0]
        param_sliders[param_name] = widgets.FloatSlider(description=param_name,
                                                            min=param_range.lo,
                                                            max=param_range.hi,
                                                            value=param_value,
                                                            layout=slider_layout,
                                                            style = slider_style,
                                                            continuous_update=False)
        param_sliders_values[param_name] = param_value
        params[param_name] = param_sliders[param_name]

    def create_ui():
    #Figure and main phase-plane axes
        # items in first vbox
        mode_selector_widget = widgets.VBox([widgets.Label('Mode Selector'), mode_selector])
        svx_widget = widgets.VBox([widgets.Label('SVX Selector'), state_variable_x])
        svy_widget = widgets.VBox([widgets.Label('SVY Selector'), state_variable_y])

        ax_widgets = widgets.VBox([reset_axes_button, sl_x_min, sl_x_max, sl_y_min, sl_y_max, 
                                    mode_selector_widget, svx_widget, svy_widget], layout=box_layout)
        sv_widgets = widgets.VBox([reset_sv_button]+list(sv_sliders.values()), layout=box_layout)
        #mod_xy_widget_list = []
        param_widgets = widgets.VBox([reset_param_button]+list(param_sliders.values()), layout=box_layout)

        if isinstance(integrator, integrators.IntegratorStochastic):
            if integrator.noise.ntau > 0.0:
                integrator.noise.configure_coloured(integrator.dt,
                                                         (1, model.nvar, 1,
                                                          model.number_of_modes))
            else:
                integrator.noise.configure_white(integrator.dt,
                                                      (1, model.nvar, 1,
                                                       model.number_of_modes))

            # RESET NOISE BUTTON
            reset_noise_button = widgets.Button(description='Reset noise strength',
                                                    disabled=False, 
                                                    layout=button_layout)
            def reset_noise(sad):
                noise_slider.value = noise_slider_valinit

            reset_noise_button.on_click(reset_noise)

            # NOISE SLIDER
            noise_slider_valinit = integrator.noise.nsig
            noise_slider = widgets.FloatSlider(description="Log Noise", 
                                                    min=-9.0,
                                                    max=1.0,
                                                    value = integrator.noise.nsig,
                                                    layout=slider_layout,
                                                    style = slider_style)

            # RESET SEED BUTTON
            reset_seed_button = widgets.Button(description='Reset random stream',
                                                    disabled=False,
                                                    layout=button_layout)
            def reset_seed(event):
                integrator.noise.reset_random_stream()

            reset_seed_button.on_click(reset_seed)

            ax_widgets.extend([reset_noise_button, noise_slider, reset_seed_button])

        items = [param_widgets, sv_widgets, ax_widgets]
        grid = widgets.GridBox(items, layout=widgets.Layout(grid_template_columns="326px 326px 326px"))
        return grid

    def printer(**plot_params):

        plot_traj_button = widgets.Button(description='Plot Trajectory')
        traj_label = widgets.Label('Trajectory Co-ordinates (Float)')
        traj_x = widgets.FloatText(placeholder='Enter X Co-ordinate (Float)', value=0.0, continuous_update=False)
        traj_y = widgets.FloatText(placeholder='Enter Y Co-ordinate (Float)', value=0.0, continuous_update=False)

        traj_box = widgets.VBox([traj_label, traj_x, traj_y], layout=box_layout)
        traj_out = widgets.Textarea(value='', placeholder='Trajectory Co-ordinates output will be shown here')

        box = widgets.HBox([plot_traj_button, traj_box, traj_out], layout=box_layout)

        display(box)

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

        
        def plot_trajectory(val):
            nonlocal ipp_fig
            nonlocal pp_ax
            nonlocal pp_splt
            nonlocal traj_x
            nonlocal traj_y

            x = traj_x.value
            y = traj_y.value
            svx_ind = model.state_variables.index(svx)
            svy_ind = model.state_variables.index(svy)

            clear_output()
            traj_out.value = f'{traj_out.value} Trajectory plotted at {x,y}\n'
            display(box)

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

            display(ipp_fig)
        
        plot_traj_button.on_click(plot_trajectory)

    ui = create_ui()
    out = interactive_output(printer, params)
    display(ui, out)
