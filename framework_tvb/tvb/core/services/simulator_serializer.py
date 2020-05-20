#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
"""

import uuid
from tvb.core.entities.file.simulator.cortex_h5 import CortexH5
from tvb.core.entities.file.simulator.simulator_h5 import SimulatorH5
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel
from tvb.core.neocom import h5


class SimulatorSerializer(object):

    @staticmethod
    def serialize_simulator(simulator, simulation_state_gid, storage_path):
        simulator_path = h5.path_for(storage_path, SimulatorH5, simulator.gid)
        with SimulatorH5(simulator_path) as simulator_h5:
            simulator_h5.store(simulator)
            if simulation_state_gid:
                simulator_h5.simulation_state.store(uuid.UUID(simulation_state_gid))

    @staticmethod
    def deserialize_simulator(simulator_gid, storage_path):
        simulator_in_path = h5.path_for(storage_path, SimulatorH5, simulator_gid)
        simulator_in = SimulatorAdapterModel()

        with SimulatorH5(simulator_in_path) as simulator_in_h5:
            simulator_in_h5.load_into(simulator_in)
            simulator_in.connectivity = simulator_in_h5.connectivity.load()
            simulator_in.stimulus = simulator_in_h5.stimulus.load()
            simulator_in.history_gid = simulator_in_h5.simulation_state.load()


        for monitor in simulator_in.monitors:
            monitor, _ = h5.load_with_links_from_dir(storage_path, monitor.gid, type(monitor))

        if simulator_in.surface:
            cortex_path = h5.path_for(storage_path, CortexH5, simulator_in.surface.gid)
            with CortexH5(cortex_path) as cortex_h5:
                simulator_in.surface.local_connectivity = cortex_h5.local_connectivity.load()
                simulator_in.surface.region_mapping_data = cortex_h5.region_mapping_data.load()
                simulator_in.surface.surface_gid = cortex_h5.surface_gid.load()

        return simulator_in
