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
An interactive Epileptor Model Visualiser.

Usage
::

    # Create and launch the interactive visualiser
    from tvb.simulator.plot.epileptor_plotter import EpileptorModelPlot
    ep = EpileptorModelPlot()
    ep.show()

"""

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

from tvb.basic.neotraits.api import HasTraits, Attr
from tvb.simulator.lab import *
from tvb.simulator.models.epileptorcodim3 import EpileptorCodim3, EpileptorCodim3SlowMod

class EpileptorModelPlot(HasTraits):
    """
    The graphical interface for visualising the epileptor model, provide controls for setting:

        - select burster class of Epileptor [List]
        - select Simulation Length [slider]
        - select Epileptor Codim Slow Model [binary]
        - select dstar[binary]
        - specify dstar value [float]
        - select modification [binary]

    """

    conn = Attr(
        field_type=connectivity.Connectivity,
        label="Connectivity",
        default=connectivity.Connectivity.from_file(),
        doc=""" The surface connectvity to be used to simulate the model. """)
    
    coupling = Attr(
        field_type=coupling.Linear,
        label="Coupling",
        default=coupling.Linear(a=np.array([0.0152])),
        doc=""" The type of coupling to be used to simulate the model. """)

    integrator = Attr(
        field_type=integrators.HeunDeterministic,
        label="Integrator",
        default=integrators.HeunDeterministic(dt=2 ** -4),
        doc=""" The integrator to be used to simulate the model. """)

    monitors = Attr(
        field_type=tuple,
        label="Monitors",
        default=(monitors.TemporalAverage(period=2 ** -2),),
        doc=""" The tuple of monitors to be used to monitor the model. """)
    
    sim_length = Attr(
        field_type=int,
        label="Simulation Length",
        default=2**10,
        doc=""" The period for which the model should be simulated. """)

    
    def __init__(self, **kwargs):
        """ Initialise based on provided keywords or their traited defaults. """

        super(EpileptorModelPlot, self).__init__(**kwargs)

        self.burster_parameters = dict()
        self.default_burster_parameters = dict()
        self.default_widget_values = dict()

    def show(self, burster_class='default'):
        """ Generate the interactive Epileptor Model Simulator with the desired burster class. """

        self.set_default_burster_parameters()
        ui = self.create_ui(burster_class = burster_class)
        self.configure_model(slow=False)
        
        if burster_class == 'default':
            self.configure_sim()
        else:
            self.update_burster_parameters(burster_class)
        
        self.plot_model()
        display(ui)

    def create_ui(self, burster_class):
        """ Create the UI for the Interactive Plotter. """

        # Default Box Layout
        self.box_layout = widgets.Layout(border='solid 1px black',
                                        margin='0px 5px 5px 0px',
                                        padding='2px 2px 2px 2px')

        # Create desired UI and add Widgets
        self.add_burster_widgets(burster_class = burster_class)
        self.add_sim_widgets()
        self.add_model_widgets()
        
        # Final control widget box
        self.control_box = widgets.HBox([self.burster_param_box, self.sim_param_box ,self.model_param_box], layout=self.box_layout)
        
        # Output Widget
        self.fig = None
        self.op = widgets.Output()
        self.op.layout = self.box_layout

        # Widget and Output Grid 
        grid = widgets.GridBox([self.control_box, self.op], layout={'grid_template_rows':'225px 625px'})
        return grid

    def set_default_burster_parameters(self):
        """ Define default Bursters and their parameters. """

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

    def add_burster_widgets(self, burster_class='default'):
        """ Add the Dropdown Widget to select desired Burster. """
        
        self.burster_choice_label = widgets.Label('Burster Choice:')
        self.burster_choice = widgets.Dropdown(options=list(self.default_burster_parameters.keys()), value=burster_class)

        self.buster_param_box_items = [self.burster_choice_label, self.burster_choice]
        self.burster_param_box = widgets.VBox(self.buster_param_box_items, layout=self.box_layout)

        self.default_widget_values['b_choice'] = self.burster_choice.value

        self.burster_choice.observe(self.update_burster_parameters, 'value')

    def add_sim_widgets(self):
        """ Add the slider for Simulation Length. """

        self.simulation_label = widgets.Label('Simulation Parameters')
        self.simulation_length_label = widgets.Label('Simulation Length: (in powers of 2)')
        self.simulation_length_slider = widgets.IntSlider(value=10, min=1, max=30, continuous_update=False)
        
        self.sim_param_box_items = [self.simulation_label, self.simulation_length_label, self.simulation_length_slider]
        self.sim_param_box = widgets.VBox(self.sim_param_box_items, layout=self.box_layout)

        self.default_widget_values['s_l'] = self.simulation_length_slider.value

        self.simulation_length_slider.observe(self.update_sim_parameters, 'value')

    def add_model_widgets(self):
        """ Add widgets to specify Model Parameters. """

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

    def add_new_burster_class(self, burster_name=None,
                             burster_value_A = None, burster_value_B = None,
                             burster_value_c = None, burster_sl=10):
        """ Function to add a new desired burster class with specific parameters. """

        if burster_name is not None and burster_value_A is not None and \
           burster_value_B is not None and burster_value_c is not None:
            self.default_burster_parameters[burster_name] = {'A':burster_value_A, 'B':burster_value_B, 'c':burster_value_c, 'sl': burster_sl}
            self.burster_choice = widgets.Dropdown(options=list(self.default_burster_parameters.keys()), value='default')

    def update_to_slow_model(self, val):
        """ Toggle Model to Epileptor Codim Slow Model. """

        if self.slowmod_checkbox.value:
            self.burster_choice.value = 'default'
            self.burster_choice.disabled = True
            self.configure_model(slow=True)

            self.simulation_length_slider.value = 14
        else:
            self.burster_choice.disabled = False
            self.configure_model(slow=False)
            self.configure_sim()
            self.plot_model()

    def update_sim_parameters(self, val):
        """ Update Simulation Length and Simulate Model. """

        self.sim_length = 2**int(self.simulation_length_slider.value)
        self.configure_sim()
        if self.slow_model:
            self.plot_slow_model()
        else:
            self.plot_model()

    def update_burster_parameters(self, val):
        """ Update the Burster Parameters of Model, configure and simulate model. """

        self.burster_parameters = self.default_burster_parameters[self.burster_choice.value]

        A = self.burster_parameters['A']
        B = self.burster_parameters['B']
        c = self.burster_parameters['c']
        
        if self.burster_choice.value != 'default':
            self.model.mu1_start=np.array([-A[1]]) 
            self.model.mu2_start=np.array([A[0]]) 
            self.model.nu_start=np.array([A[2]])
            self.model.mu1_stop=np.array([-B[1]]) 
            self.model.mu2_stop=np.array([B[0]]) 
            self.model.nu_stop=np.array([B[2]]) 
            self.model.c=np.array([c])
            self.simulation_length_slider.value = int(self.burster_parameters['sl'])
        else:
            self.simulation_length_slider.value = 10
            self.configure_model(slow=False)
        
        self.configure_sim()
        self.plot_model()

    def update_model_parameters(self, val):
        """ Update Model Parameters, configure and simulate Model. """

        if self.dstar_checkbox.value or self.mod_checkbox.value:
            self.burster_choice.disabled = True
            self.model.dstar = np.array([self.dstar_value.value])
            self.model.modification = np.array([self.mod_checkbox.value])
            self.configure_sim()
            self.plot_model()
        else:
            self.burster_choice.disabled = False
            self.configure_model()
            self.configure_sim()
            self.plot_model()


    def configure_model(self, slow=False):
        """ Configure Model according to desired choice. (Normal/Slow) """

        if slow:
            self.slow_model = True
            self.model = EpileptorCodim3SlowMod()
        else:
            self.slow_model = False
            self.model = EpileptorCodim3(variables_of_interest=['x', 'y', 'z'])

    def configure_sim(self):
        """ Configure Simulation Parameters and run Simulation. """

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
        """ Plot Normal Model Simulation Output. """
        self.op.clear_output(wait=True)
        with plt.ioff():
            if not self.fig:
                self.fig = plt.figure(figsize=(9,5))
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
            with self.op:
                display(self.fig.canvas)

    def plot_slow_model(self):
        """ Plot Slow Model Simulation Output. """

        self.fig.clear()
        
        plt.plot(self.tavg_time, self.tavg_data[:, 0, 0, 0], label='x')
        plt.plot(self.tavg_time, self.tavg_data[:, 1, 0, 0], label='z')
        plt.legend()
        plt.grid(True)
        plt.xlabel('Time (ms)')
        plt.ylabel("Temporal average")