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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tvb.simulator.lab import *
import ipywidgets as widgets
from IPython.display import display
from tvb.simulator.lab import *
from tvb.simulator.models.epileptorcodim3 import EpileptorCodim3, EpileptorCodim3SlowMod

class EpileptorModelPlot:
    def __init__(self,
                conn = connectivity.Connectivity.from_file(), 
                coupling = coupling.Linear(a=np.array([0.0152])),
                integrator = integrators.HeunDeterministic(dt=2 ** -4),
                monitors = (monitors.TemporalAverage(period=2 ** -2),),
                sim_length = 2**10):
        
        self.conn = conn
        self.coupling = coupling
        self.integrator = integrator
        self.monitors = monitors
        self.sim_length = sim_length
        self.simulator_bursters = ['default','c0']
        self.burster_parameters = dict()
        self.default_burster_parameters = dict()
        self.default_widget_values = dict()


    def show(self):
        self.set_default_burster_parameters()
        ui = self.create_ui()
        self.configure_model(slow=False)
        self.configure_sim()
        
        self.plot_model()
        display(ui)

    def create_ui(self):
        self.box_layout = widgets.Layout(border='solid 1px black',
                                        margin='0px 5px 5px 0px',
                                        padding='2px 2px 2px 2px')

        self.add_burster_widgets()
        self.add_sim_widgets()
        self.add_model_widgets()
        
        self.control_box = widgets.HBox([self.burster_param_box, self.sim_param_box, self.model_param_box], layout=self.box_layout)
        

        op = widgets.Output()
        with op:
            self.fig = plt.figure(figsize=(9,5))
            self.fig1 = self.fig.add_subplot(121)
            self.fig2 = self.fig.add_subplot(122, projection='3d')
        
        self.output_box = op
        self.output_box.layout = self.box_layout

        grid = widgets.GridBox([self.control_box, self.output_box], layout={'grid_template_rows':'225px 625px'})
        return grid

    def set_default_burster_parameters(self):
        self.default_burster_parameters['default'] = {'A':None, 'B':None, 'c':None}
        self.default_burster_parameters['c0'] = {'A':[0.2649, -0.05246, 0.2951], 'B':[0.2688, 0.05363, 0.2914], 'c':0.001, 'sl':11}
        self.default_burster_parameters['c2s'] = {'A':[0.3448,0.02285,0.2014], 'B':[0.3351,0.07465,0.2053], 'c':0.001, 'sl':10}
        self.default_burster_parameters['c3s'] = {'A':[0.2552,-0.0637,0.3014], 'B':[0.3496,0.0795,0.1774], 'c':0.0004, 'sl':13}
        self.default_burster_parameters['c10s'] = {'A':[0.3448,0.0228,0.2014], 'B':[0.3118,0.0670,0.2415], 'c':0.00005, 'sl':14}
        self.default_burster_parameters['c11s'] = {'A':[0.3131,-0.06743,0.2396], 'B':[0.3163,0.06846,0.2351], 'c':0.00004, 'sl':15}
        self.default_burster_parameters['c2b'] = {'A':[0.3216,0.0454,-0.2335], 'B':[0.285,0.05855,-0.2745], 'c':0.004, 'sl':10}
        self.default_burster_parameters['c4b'] = {'A':[0.1871,-0.02512,-0.3526], 'B':[0.2081,-0.01412,-0.3413], 'c':0.008, 'sl':10}
        self.default_burster_parameters['c14b'] = {'A':[0.3216,0.0454,-0.2335], 'B':[0.106,0.005238,-0.3857], 'c':0.002, 'sl':12}
        self.default_burster_parameters['c16b'] = {'A':[0.04098,-0.07373,-0.391], 'B':[-0.01301,-0.03242,-0.3985], 'c':0.004, 'sl':10}

    def add_burster_widgets(self):
        self.burster_choice_label = widgets.Label('Burster Choice:')
        self.burster_choice = widgets.Dropdown(options=list(self.default_burster_parameters.keys()), value='default')
        
        # self.burster_label = widgets.Label('Burster Parameters')
        # self.burster_A = widgets.Text(placeholder='Enter the value of burster parameter "A"', value = None, continuous_update=False)
        # self.burster_B = widgets.Text(placeholder='Enter the value of burster parameter "B"', value = None, continuous_update=False)
        # self.burster_c = widgets.Text(placeholder='Enter the value of burster parameter "c"', value = None, continuous_update=False)
        
        # self.buster_param_box_items = [self.burster_choice_label, self.burster_choice, self.burster_label, self.burster_A, self.burster_B, self.burster_c]
        # self.burster_param_box = widgets.VBox(self.buster_param_box_items, layout=self.box_layout)
        self.buster_param_box_items = [self.burster_choice_label, self.burster_choice]
        self.burster_param_box = widgets.VBox(self.buster_param_box_items, layout=self.box_layout)

        self.default_widget_values['b_choice'] = self.burster_choice.value
        # self.default_widget_values['b_A'] = self.burster_A.value
        # self.default_widget_values['b_B'] = self.burster_B.value
        # self.default_widget_values['b_c'] = self.burster_c.value

        self.burster_choice.observe(self.update_burster_parameters, 'value')
        # self.burster_A.observe(self.update_burster_parameters, 'value')
        # self.burster_B.observe(self.update_burster_parameters, 'value')
        # self.burster_c.observe(self.update_burster_parameters, 'value')

    def add_sim_widgets(self):
        self.simulation_label = widgets.Label('Simulation Parameters')
        self.simulation_length_label = widgets.Label('Simulation Length: (in powers of 2)')
        self.simulation_length_slider = widgets.IntSlider(value=10, min=10, max=15, continuous_update=False)
        
        self.sim_param_box = widgets.VBox([self.simulation_length_label, self.simulation_length_slider], layout=self.box_layout)

        self.default_widget_values['s_l'] = self.simulation_length_slider.value

        self.simulation_length_slider.observe(self.update_sim_parameters, 'value')

    def add_model_widgets(self):
        self.model_parameter_label = widgets.Label('Model Parameters')
        self.slowmod_checkbox = widgets.Checkbox(description='Slow Transition', value=False)
        self.dstar_checkbox = widgets.Checkbox(description='dstar', value=False)
        self.dstar_value = widgets.FloatText(description='dstar', value=0.0, continuous_update=False)
        self.mod_checkbox = widgets.Checkbox(description='Modification', value=False)
        
        self.model_param_items = [self.model_parameter_label,self.slowmod_checkbox,self.dstar_checkbox,self.dstar_value, self.mod_checkbox]
        self.model_param_box = widgets.VBox(self.model_param_items, layout=self.box_layout)

        self.slowmod_checkbox.observe(self.update_to_slow_model,'value')
        self.dstar_checkbox.observe(self.update_model_parameters,'value')
        self.dstar_value.observe(self.update_model_parameters,'value')
        self.mod_checkbox.observe(self.update_model_parameters,'value')

    def add_new_burster_class(self, burster_name=None, burster_value_A = None, burster_value_B = None, burster_value_c = None, burster_sl=10):
        if burster_name is not None and burster_value_A is not None and burster_value_B is not None and burster_value_c is not None:
            self.default_burster_parameters[burster_name] = {'A':burster_value_A, 'B':burster_value_B, 'c':burster_value_c, 'sl': burster_sl}
            self.burster_choice = widgets.Dropdown(options=list(self.default_burster_parameters.keys()), value='default')

    def reset_params(self):
        self.burster_choice.value = self.default_widget_values['b_choice']
        self.burster_A.value = self.default_widget_values['b_A']
        self.burster_B.value = self.default_widget_values['b_B']
        self.burster_c.value = self.default_widget_values['b_c']

        self.simulation_length_slider = self.default_widget_values['s_l']

    def trigger_model_params(self, disabled=True):
        self.burster_choice.disabled = disabled
        self.burster_A.disabled = disabled
        self.burster_B.disabled = disabled
        self.burster_c.disabled = disabled

    def update_to_slow_model(self, val):
        if self.slowmod_checkbox.value:
            self.trigger_model_params(True)
            self.configure_model(slow=True)

            self.simulation_length_slider.value = 14
        else:
            self.trigger_model_params(False)
            self.configure_model(slow=False)
            self.configure_sim()
            self.plot_model()

    def update_sim_parameters(self, val):
        self.sim_length = 2**int(self.simulation_length_slider.value)
        self.configure_sim()
        if self.slow_model:
            self.plot_slow_model()
        else:
            self.plot_model()

    def update_burster_parameters(self, val):
        self.burster_parameters = self.default_burster_parameters[self.burster_choice.value]
        
        # if self.burster_A.value:
        #     self.burster_parameters['A'] = self.burster_A.value
        # if self.burster_B.value:
        #     self.burster_parameters['B'] = self.burster_B.value
        # if self.burster_c.value:
        #     self.burster_parameters['c'] = self.burster_c.value

        A = self.burster_parameters['A']
        B = self.burster_parameters['B']
        c = self.burster_parameters['c']
        
        self.model.mu1_start=np.array([-A[1]]) 
        self.model.mu2_start=np.array([A[0]]) 
        self.model.nu_start=np.array([A[2]])
        self.model.mu1_stop=np.array([-B[1]]) 
        self.model.mu2_stop=np.array([B[0]]) 
        self.model.nu_stop=np.array([B[2]]) 
        self.model.c=np.array([c])
        self.simulation_length_slider.value = int(self.burster_parameters['sl'])
        
        self.configure_sim()
        self.plot_model()

    def update_model_parameters(self, val):
        if self.dstar_checkbox.value or self.mod_checkbox.value:
            # RESET PARAMS TO DEFAULT AND DISABLE PARAMS
            self.reset_params()
            self.trigger_model_params(disabled=True)
            self.model.dstar = np.array([self.dstar_value.value])
            self.model.modification = np.array([self.mod_checkbox.value])
            self.configure_sim()
            self.plot_model()
        else:
            self.trigger_model_params(disabled=False)
            self.configure_model()
            self.configure_sim()
            self.plot_model()


    def configure_model(self, slow=False):
        if slow:
            self.slow_model = True
            self.model = EpileptorCodim3SlowMod()
        else:
            self.slow_model = False
            self.model = EpileptorCodim3(variables_of_interest=['x', 'y', 'z'])

    def configure_sim(self):
        self.sim = simulator.Simulator(
            model=self.model,
            connectivity=self.conn,
            coupling=self.coupling,
            integrator=self.integrator,
            monitors=self.monitors,
            simulation_length=self.sim_length)
        
        self.sim = self.sim.configure()

        (self.tavg_time, self.tavg_data), = self.sim.run()

    def plot_model(self):
        self.fig.clear()
        self.fig1 = self.fig.add_subplot(121)
        self.fig2 = self.fig.add_subplot(122, projection='3d')

        self.fig1.plot(self.tavg_time, self.tavg_data[:, 0, 0, 0], label='x')
        self.fig1.plot(self.tavg_time, self.tavg_data[:, 2, 0, 0], label='z')
        
        self.fig1.legend()
        self.fig1.grid(True)
        self.fig1.set_xlabel('Time (ms)')
        self.fig1.set_ylabel("Temporal average")

        self.fig2.plot(self.tavg_data[:, 0, 0, 0], self.tavg_data[:, 1, 0, 0], self.tavg_data[:, 2, 0, 0])
        
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')

    def plot_slow_model(self):
        self.fig.clear()
        
        plt.plot(self.tavg_time, self.tavg_data[:, 0, 0, 0], label='x')
        plt.plot(self.tavg_time, self.tavg_data[:, 1, 0, 0], label='z')
        plt.legend()
        plt.grid(True)
        plt.xlabel('Time (ms)')
        plt.ylabel("Temporal average")