# -*- coding: utf-8 -*-
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
import os
import uuid
import typing
from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.basic.neotraits.api import HasTraits
from tvb.basic.profile import TvbProfile
from tvb.core.neocom import h5


class HPCSimulatorAdapter(SimulatorAdapter):
    OUTPUT_FOLDER = 'output'

    @staticmethod
    def load_traited_by_gid(data_gid, dt_class=None):
        # type: (uuid.UUID, typing.Type[HasTraits]) -> HasTraits
        """
        Load a generic HasTraits instance, specified by GID.
        """
        return h5.load_from_dir(TvbProfile.current.hpc.HPC_INPUT_FOLDER, data_gid, dt_class=dt_class)

    def _try_load_region_mapping(self):
        return None, None

    def _is_group_launch(self):
        """
        Return true if this adapter is launched from a group of operations
        """
        # TODO: treat this check
        return False

    def _get_output_path(self):
        output_path = os.path.join(self.storage_path, self.OUTPUT_FOLDER)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        return output_path
