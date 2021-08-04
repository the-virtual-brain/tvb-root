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
from tvb.simulator.lab import *
import ipywidgets as widgets
from IPython.display import display

class HeadPlotter3D:
    def __init__(self):
        pass

    def display_source_sensor_geometry(self, meg_sensors = None, eeg_sensors = None, conn = None, skin = None):
        # type: (sensors.SensorsMEG, sensors.SensorsEEG, connectivity.Connectivity, surfaces.SkinAir) -> None
        """
        :param meg_sensors: Optional sensors.SensorsMEG instance. When none, we will try loading a default
        :param eeg_sensors: Optional sensors.SensorsEEG instance. When none, we will try loading a default
        :param conn: Optional connectivity.Connectivity instance. When none, we will try loading a default
        :param skin: Optional surfaces.SkinAir instance. When none, we will try loading a default
        """

        if meg_sensors is None:
            meg_sensors = sensors.SensorsMEG.from_file()
        if eeg_sensors is None:
            eeg_sensors = sensors.SensorsEEG.from_file()
        if conn is None:
            conn = connectivity.Connectivity.from_file()
        if skin is None:
            skin = surfaces.SkinAir.from_file()
        
        # Configure Skin Surface
        skin.configure()

        # Configure EEG Sensors
        eeg_sensors.configure()

        # WIDGET Controls and Layout
        box_layout = widgets.Layout(border='solid 1px black', margin='3px 3px 3px 3px', padding='2px 2px 2px 2px')
        params = dict()
        roi_checkbox = widgets.Checkbox(description="Show ROI Centers", value=False)
        eeg_checkbox = widgets.Checkbox(description="Show EEG Sensors", value=False)
        meg_checkbox = widgets.Checkbox(description="Show MEG Sensors", value=False)

        control_box = widgets.HBox([roi_checkbox,eeg_checkbox,meg_checkbox], layout=box_layout)

        # Connecting widgets with plot parameters
        params['ROI'] = roi_checkbox
        params['EEG'] = eeg_checkbox
        params['MEG'] = meg_checkbox        

        # Plotter Function
        def plot(**plot_params):
            ax = plt.subplot(111, projection='3d')

            # Plot boundary surface
            x, y, z = skin.vertices.T
            ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='none')

            if plot_params['ROI']:
                # ROI centers as black circles
                x, y, z = conn.centres.T
                ax.plot(x, y, z, 'ko')
            
            if plot_params['EEG']:
                # EEG sensors as blue x's
                x, y, z = eeg_sensors.sensors_to_surface(skin).T
                ax.plot(x, y, z, 'bx')

            if plot_params['MEG']:
                # MEG sensors as red +'s
                x, y, z = meg_sensors.locations.T
                ax.plot(x, y, z, 'r+')
        plt.figure()

        out = widgets.interactive_output(plot, params)
        display(control_box,out)

    def display_surface_local_connectivity(self, ctx=None, loc_conn=None):
        # type: (cortex.Cortex, local_connectivity.LocalConnectivity) -> None
        """
        :param cortex: Optional cortex.Cortex instance. When none, we will try loading a default
        :param loc_conn: Optional. If None, and
        """
        # Start by configuring the cortical surface
        if ctx is None:
            ctx = cortex.Cortex.from_file()
        if ctx.local_connectivity is None:
            if loc_conn is None:
                loc_conn = local_connectivity.LocalConnectivity(cutoff=20.0, surface=ctx.region_mapping_data.surface)
            loc_conn.equation.parameters['sigma'] = 10.0
            loc_conn.equation.parameters['amp'] = 1.0
            ctx.local_connectivity = loc_conn
        ctx.coupling_strength = np.array([0.0115])        
        ctx.configure()
        
        # plot 
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        x, y, z = ctx.vertices.T
        ax.plot_trisurf(x, y, z, triangles=ctx.triangles, alpha=0.1, edgecolor='none')