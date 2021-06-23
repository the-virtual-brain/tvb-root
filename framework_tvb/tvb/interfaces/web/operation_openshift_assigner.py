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
"""
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""
import random
from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep

import requests
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_operation import Operation
from tvb.core.entities.storage import dao
from tvb.core.services.kube_service import KubeService

log = get_logger(__name__)

if __name__ == '__main__':
    TvbProfile.set_profile(TvbProfile.WEB_PROFILE, True)
    if TvbProfile.current.web.OPENSHIFT_DEPLOY:
        log.info("Start operation assigner")
        while True:
            sleep(TvbProfile.current.OPERATIONS_BACKGROUND_JOB_INTERVAL)
            operations = dao.get_generic_entity(Operation, True, "queue_full")
            log.info("Found {} operations with the queue full flag set.".format(len(operations)))
            if len(operations) == 0:
                continue
            pods, auth_header = KubeService.get_pods(TvbProfile.current.web.OPENSHIFT_PROCESSING_OPERATIONS_APPLICATION)
            if pods:
                random.shuffle(pods)
                pods_no = len(pods)
                operations.sort(key=lambda l_operation: l_operation.id)
                for index, operation in enumerate(operations[0:TvbProfile.current.MAX_THREADS_NUMBER*pods_no]):
                    pod_ip = pods[index % pods_no]['ip']
                    log.info("Notify pod: {}".format(pod_ip))
                    url_pattern = "http://{}:{}/kube/start_operation_pod/{}"
                    requests.get(url=url_pattern.format(pod_ip, TvbProfile.current.web.SERVER_PORT, operation.id),
                                 headers=auth_header)
    else:
        log.info("Openshift deploy is not enabled.")
