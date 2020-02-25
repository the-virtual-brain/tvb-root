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
import sys
from tvb.adapters.simulator.hpc_simulator_adapter import HPCSimulatorAdapter
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.core.services.simulator_serializer import SimulatorSerializer

if __name__ == '__main__':
    TvbProfile.set_profile(TvbProfile.WEB_PROFILE)
    TvbProfile.current.hpc.IS_HPC_RUN = True


def do_operation_launch(simulator_gid, available_disk_space):
    log = get_logger('tvb.core.operation_hpc_launcher')
    try:
        log.info("Preparing HPC launch for simulation with id={}".format(simulator_gid))
        populate_datatypes_registry()
        log.info("Current TVB profile has HPC run=: {}".format(TvbProfile.current.hpc.IS_HPC_RUN))
        input_folder = os.getcwd()
        log.info("Current wdir is: {}".format(input_folder))
        view_model = SimulatorSerializer().deserialize_simulator(simulator_gid, input_folder)
        adapter_instance = HPCSimulatorAdapter(input_folder)
        result_msg, nr_datatypes = adapter_instance._prelaunch(None, None, available_disk_space, view_model)

    except Exception as excep:
        log.error("Could not execute operation {}".format(str(sys.argv[1])))
        log.exception(excep)


if __name__ == '__main__':
    simulator_gid = sys.argv[1]
    available_disk_space = sys.argv[2]

    do_operation_launch(simulator_gid, available_disk_space)
