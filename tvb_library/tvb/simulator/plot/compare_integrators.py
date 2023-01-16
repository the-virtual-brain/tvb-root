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
An interactive visualiser to compare different Integrators.

Usage
::

    #Create and launch the interactive visualiser
    from tvb.simulator.plot.compare_integrators import CompareIntegrators
    ci = CompareIntegrators()
    ci.show()

"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

from tvb.basic.neotraits.api import HasTraits, Attr
from tvb.simulator.lab import *
import tvb.simulator.models as models_module
import tvb.simulator.coupling as coupling_module

import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import clear_output

FONT_SIZE = "font.size"

default_base_dt = 0.1
default_var_order_dt = 5.0

default_methods = [
    (integrators.EulerDeterministic, default_base_dt),
    (integrators.HeunDeterministic, 2 * default_base_dt),
    (integrators.Dop853, default_var_order_dt),
    (integrators.Dopri5, default_var_order_dt),
    (integrators.RungeKutta4thOrderDeterministic, 4 * default_base_dt),
    (integrators.VODE, default_var_order_dt),
]


class CompareIntegrators(HasTraits):
    """
    The graphical interface for comparing different integrators 
    provide controls for setting:

        - how to compare integrators
        - which integrators to compare

    """

    conn = Attr(
        field_type=connectivity.Connectivity,
        label="Connectivity",
        default=connectivity.Connectivity.from_file(),
        doc=""" The connectivity required to compare integrators. """)

    model = Attr(
        field_type=models_module.Model,
        label="Model",
        default=models.Generic2dOscillator(a=np.array([0.1])),
        doc=""" The model required to compare integrators. """)

    coupling = Attr(
        field_type=coupling_module.Coupling,
        label="Coupling",
        default=coupling.Linear(a=np.array([0.0])),
        doc=""" The desired coupling required to compare integrators. """)

    monitors = Attr(
        field_type=tuple,
        label="Monitors",
        default=(monitors.TemporalAverage(period=5.0),),
        doc=""" The monitors required to monitor the compared integrators. """)

    methods = Attr(
        field_type=list,
        label="Integrators to compare",
        default=default_methods,
        doc=""" The desired integrators to compare. Pass it in a list. E.g. [(Integrator, default_dt), ]. """)

    def __init__(self, **kwargs):
        """ Initialise based on provided keywords or their traited defaults. """

        super(CompareIntegrators, self).__init__(**kwargs)

        self.plot_params = dict()

    def create_ui(self, comparison):
        """ Create Interactive UI to compare integrators. """

        self.fig_size = (13, 13)
        self.select_comparison_label = widgets.Label('Compare: ')
        self.select_comparison = widgets.Dropdown(options=['Default', 'Pairwise', 'dt Growth'], value=comparison)
        controls = widgets.HBox([self.select_comparison_label, self.select_comparison])
        self.plot_params['comparison'] = self.select_comparison

        output = widgets.VBox([controls])
        return output

    def show(self, comparison='Default'):
        """ Generate interactive Compare Integrators Figure. """

        ui = self.create_ui(comparison)

        def plotter(**plot_params):
            val = plot_params['comparison']
            if val == 'dt Growth':
                self.grow_dt()
            elif val == 'Pairwise':
                self.compare_pairwise()
            else:
                self.compare()

        out = widgets.interactive_output(plotter, self.plot_params)
        display(ui, out)

    def compare(self, sim_length=1000.0):
        """ Compare Integrators Simulation. """

        clear_output()
        plt.figure(figsize=self.fig_size)
        plt.rcParams[FONT_SIZE] = "10"
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
            plt.title(method.__name__, wrap=True)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
            plt.grid(True)

    def compare_pairwise(self, sim_length=200.0):
        """ Compare Integrators Simulation Pairwise. """

        clear_output()
        raws, names, t = self.get_simulation_data_for_integrators(sim_length)
        n_raw = len(raws)

        plt.figure(figsize=self.fig_size)
        plt.rcParams[FONT_SIZE] = "6"
        plt.autoscale(True)

        for values in itertools.product(enumerate(zip(raws, names)), repeat=2):
            i, (raw_i, name_i) = values[0]
            j, (raw_j, name_j) = values[1]
            plt.subplot(n_raw, n_raw, i * n_raw + j + 1)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)
            if raw_i.shape[0] != t.shape[0] or raw_i.shape[0] != raw_j.shape[0]:
                continue

            # execute these steps only if there is something to plot
            plt.title(None if i else name_j, wrap=True)
            plt.ylabel(None if j else name_i, wrap=True)
            plt.grid(True)

            if i == j:
                plt.plot(t, raw_i[:, 0, :, 0], 'k', alpha=0.1)
            else:
                plt.semilogy(t, (abs(raw_i - raw_j) / raw_i.ptp())[:, 0, :, 0], 'k', alpha=0.1)
                plt.ylim(np.exp(np.r_[-30, 0]))

        plt.tight_layout()

    def grow_dt(self, sim_length=1200.0):
        """ Compare single Integrator with Dt Growth. """

        clear_output()
        dts = [float(10 ** e) for e in np.r_[-2:0:10j]]

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
        plt.rcParams[FONT_SIZE] = "10"

        for i, dt in enumerate(dts):
            plt.subplot(len(dts) // 3, 3 + 1, i + 1)
            plt.autoscale(True)
            if i == 0:
                dat = raws[i]
            else:
                dat = np.log10((abs(raws[i] - raws[0]) / raws[0].ptp()))
            plt.plot(t, dat[:, 0, :, 0], 'k', alpha=0.1)

    def get_simulation_data_for_integrators(self, sim_length):
        raws, names = [], []
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
            names.append(method.__name__)

        return raws, names, t
