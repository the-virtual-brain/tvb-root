# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
"""
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.config import SIMULATOR_CLASS, SIMULATOR_MODULE
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.load import get_class_by_name
from tvb.core.entities.storage import dao
from tvb.core.services.backend_clients.backend_client import BackendClient
from tvb.core.services.backend_clients.cluster_scheduler_client import ClusterSchedulerClient
from tvb.core.services.backend_clients.hpc_scheduler_client import HPCSchedulerClient
from tvb.core.services.backend_clients.standalone_client import StandAloneClient
from tvb.core.services.exceptions import InvalidSettingsException

LOGGER = get_logger(__name__)


class BackendClientFactory(object):

    @staticmethod
    def _get_backend_client(adapter_instance):
        # type: (ABCAdapter) -> BackendClient

        # For the moment run only simulations on HPC
        if TvbProfile.current.hpc.IS_HPC_RUN and type(adapter_instance) is get_class_by_name(
                "{}.{}".format(SIMULATOR_MODULE, SIMULATOR_CLASS)):
            if not TvbProfile.current.hpc.CAN_RUN_HPC:
                raise InvalidSettingsException("We can not enable HPC run. Most probably pyunicore is not installed!")
            # Return an entity capable to submit jobs to HPC.
            return HPCSchedulerClient()
        if TvbProfile.current.cluster.IS_DEPLOY:
            # Return an entity capable to submit jobs to the cluster.
            return ClusterSchedulerClient()
        # Return a thread launcher.
        return StandAloneClient()

    @staticmethod
    def execute(operation_id, user_name_label, adapter_instance):
        backend_client = BackendClientFactory._get_backend_client(adapter_instance.stored_adapter)
        backend_client.execute(operation_id, user_name_label, adapter_instance)

    @staticmethod
    def stop_operation(operation_id):
        operation = dao.get_operation_by_id(operation_id)
        algorithm = operation.algorithm
        backend_client = BackendClientFactory._get_backend_client(algorithm)
        return backend_client.stop_operation(operation_id)
