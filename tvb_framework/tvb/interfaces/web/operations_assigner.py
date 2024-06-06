# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2024, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Entry point for a POD on kubernetes cloud, to monitor submitted ops and launch an executor.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

from time import sleep
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_operation import Operation
from tvb.core.entities.storage import dao
from tvb.storage.kube.kube_notifier import KubeNotifier

LOGGER = get_logger("tvb.interfaces.web.operations_assigner")

if __name__ == '__main__':

    TvbProfile.set_profile(TvbProfile.WEB_PROFILE, True)

    if TvbProfile.current.web.IS_CLOUD_DEPLOY:
        LOGGER.info("Starting operation assigner ...")

        while True:
            sleep(TvbProfile.current.OPERATIONS_BACKGROUND_JOB_INTERVAL)
            operations = dao.get_generic_entity(Operation, True, "queue_full")
            LOGGER.info("Found {} operations with the queue full flag set.".format(len(operations)))

            if len(operations) == 0:
                continue

            auth_header = KubeNotifier.get_authorization_header()
            operations.sort(key=lambda l_operation: l_operation.id)

            for index, operation in enumerate(operations[0:TvbProfile.current.MAX_THREADS_NUMBER]):
                KubeNotifier.do_rest_call_to_pod(TvbProfile.current.web.CLOUD_APP_EXEC_NAME,
                                                 'start_operation_pod', operation.id)

    else:
        LOGGER.info("Cloud deploy is not enabled (in ~/.tvb.configuration), This process will stop!")
