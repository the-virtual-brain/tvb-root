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

from tvb.adapters.simulator.simulator_adapter import SimulatorAdapter
from tvb.basic.logger.builder import get_logger
from tvb.core.neocom.h5 import DirLoader
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, InvalidInputException, ServiceException
from tvb.simulator.simulator import Simulator


class SimulationFacade:
    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.simulator_service = SimulatorService()
        self.project_service = ProjectService()

    def launch_simulation(self, current_user_id, zip_directory, project_gid):
        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException()

        try:
            simulator_h5_name = DirLoader(zip_directory, None).find_file_for_has_traits_type(Simulator)
            simulator_file = os.path.join(zip_directory, simulator_h5_name)
        except IOError:
            raise InvalidInputException('No Simulator h5 file found in the archive')

        try:
            simulator_algorithm = AlgorithmService().get_algorithm_by_module_and_class(SimulatorAdapter.__module__,
                                                                                       SimulatorAdapter.__name__)
            simulation = self.simulator_service.prepare_simulation_on_server(user_id=current_user_id,
                                                                             project=project,
                                                                             algorithm=simulator_algorithm,
                                                                             zip_folder_path=zip_directory,
                                                                             simulator_file=simulator_file)
            return simulation.gid
        except Exception as excep:
            self.logger.error(excep, exc_info=True)
            raise ServiceException(str(excep))
