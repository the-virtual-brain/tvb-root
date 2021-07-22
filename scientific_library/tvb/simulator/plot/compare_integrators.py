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

from IPython.core.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
from tvb.simulator.lab import *
from IPython.display import display
import ipywidgets as widgets

default_base_dt = 0.1
default_var_order_dt = 5.0

default_methods = [
    (integrators.EulerDeterministic, default_base_dt),
    (integrators.HeunDeterministic, 2*default_base_dt),
    (integrators.Dop853, default_var_order_dt),
    (integrators.Dopri5, default_var_order_dt),
    #(integrators.RungeKutta4thOrderDeterministic, 4*default_base_dt),
    (integrators.VODE, default_var_order_dt),
]

class CompareIntegrators:
    def __init__(self, 
                methods = default_methods, 
                conn = connectivity.Connectivity.from_file(),
                model = models.Generic2dOscillator(a=np.array([0.1])),
                coupling = coupling.Linear(a=np.array([0.0])),
                monitors = (monitors.TemporalAverage(period=5.0),)):
        
        self.methods = methods
        self.conn = conn
        self.fig_size = (9,9)
        self.model = model
        self.coupling = coupling
        self.monitors = monitors
        self.plot_params = dict()
    
    def create_ui(self):
        self.select_comparison_label = widgets.Label('Compare: ')
        self.select_comparison = widgets.Dropdown(options = ['Default', 'Pairwise', 'dt Growth'], default='Default')
        controls = widgets.HBox([self.select_comparison_label, self.select_comparison])
        self.plot_params['comparison'] = self.select_comparison

        output = widgets.VBox([controls])
        return output
    
    def show(self):
        ui = self.create_ui()
        
        def plotter(**plot_params):
            val = plot_params['comparison']
            if val == 'dt Growth':
                self.grow_dt()
            elif val == 'Pairwise':
                self.compare_pairwise()
            else:
                self.compare()
        
        out = widgets.interactive_output(plotter, self.plot_params)
        display(ui,out)


    def compare(self, sim_length=1000.0):
        clear_output()
        plt.figure(figsize=self.fig_size)
        plt.rcParams["font.size"] = "10"
        for i, (method, dt) in enumerate(self.methods):
            np.random.seed(42)
            sim = simulator.Simulator(
                connectivity=self.conn,
                model=self.model,
                coupling=self.coupling,
                integrator=method(dt=dt),
                monitors=self.monitors,
                simulation_length=sim_length,
            ).configure()
            (t, raw), = sim.run()
            
            if i == 0:
                euler_raw = raw
            else:
                if raw.shape[0] != euler_raw.shape[0]:
                    continue
                raw = abs(raw - euler_raw) / euler_raw.ptp() * 100.0
            
            plt.subplot(3, 2, i + 1)
            plt.autoscale(True)
            plt.plot(t, raw[:, 0, :, 0], 'k', alpha=0.1)
            if i > 0:
                plt.ylabel('% diff')
                plt.plot(t, raw[:, 0, :, 0].mean(axis=1), 'k', linewidth=3)
            plt.title(method._ui_name, wrap=True)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
            plt.grid(True)

    def compare_pairwise(self, sim_length=200.0):
        clear_output()
        raws = []
        names = []
        for i, (method, dt) in enumerate(self.methods):
            np.random.seed(42)
            sim = simulator.Simulator(
                connectivity=self.conn,
                model=self.model,
                coupling=self.coupling,
                integrator=method(dt=dt),
                monitors=self.monitors,
                simulation_length=sim_length,
            ).configure()
            (t, raw), = sim.run()
            raws.append(raw)
            names.append(method._ui_name)

        n_raw = len(raws)
        plt.figure(figsize=self.fig_size)
        plt.rcParams["font.size"] = "6"
        for i, (raw_i, name_i) in enumerate(zip(raws, names)):
            for j, (raw_j, name_j) in enumerate(zip(raws, names)):
                plt.subplot(n_raw, n_raw, i*n_raw + j + 1)
                plt.autoscale(True)
                if raw_i.shape[0] != t.shape[0] or raw_i.shape[0] != raw_j.shape[0]:
                    continue
                if i == j:
                    plt.plot(t, raw_i[:, 0, :, 0], 'k', alpha=0.1)
                else:
                    plt.semilogy(t, (abs(raw_i - raw_j) / raw_i.ptp())[:, 0, :, 0], 'k', alpha=0.1)
                    plt.ylim(np.exp(np.r_[-30, 0]))
                
                plt.grid(True)
                if i==0:
                    plt.title(name_j, wrap = True)
                if j==0:
                    plt.ylabel(name_i, wrap=True)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)
            
            if i == 0:
                euler_raw = raw
            else:
                raw = abs(raw - euler_raw) / euler_raw.ptp() * 100.0
        plt.tight_layout()

    def grow_dt(self, sim_length=1200.0):
        clear_output()
        dts = [float(10**e) for e in np.r_[-2:0:10j]]

        raws = []
        for i, dt in enumerate(dts):
            np.random.seed(42)
            sim = simulator.Simulator(
                connectivity=self.conn,
                model=self.model,
                coupling=self.coupling,
                integrator=integrators.VODE(dt=dt),
                monitors=(monitors.TemporalAverage(period=1.0),),
                simulation_length=sim_length,
            ).configure()
            (t, raw), = sim.run()
            t = t[:1000]
            raw = raw[:1000]
            raws.append(raw)
        
        plt.figure(figsize=self.fig_size)
        plt.rcParams["font.size"] = "10"
    
        for i, dt in enumerate(dts):
            plt.subplot(len(dts)//3, 3+1, i + 1)
            plt.autoscale(True)
            if i == 0:
                dat = raws[i]
            else:
                dat = np.log10((abs(raws[i] - raws[0]) / raws[0].ptp()))
            plt.plot(t, dat[:, 0, :, 0], 'k', alpha=0.1)