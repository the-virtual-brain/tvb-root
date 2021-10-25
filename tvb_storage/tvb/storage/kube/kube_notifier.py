# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
"""
Service layer used for kubernetes calls.

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""
from concurrent.futures.thread import ThreadPoolExecutor
from subprocess import Popen, PIPE

import requests
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile

LOGGER = get_logger(__name__)


class KubeNotifier:
    @staticmethod
    def get_pods(application):
        sa_token = Popen(['cat', '/var/run/secrets/kubernetes.io/serviceaccount/token'], stdout=PIPE,
                         stderr=PIPE).stdout.read().decode()

        auth_header = {"Authorization": "Bearer {}".format(sa_token)}
        openshift_pods = None
        try:
            response = KubeNotifier.fetch_endpoints(auth_header, application)
            openshift_pods = response.json()['subsets'][0]['addresses']
        except Exception as e:
            LOGGER.error("Failed to retrieve openshift pods for application {}".format(application), e)
        return openshift_pods, auth_header

    @staticmethod
    def notify_pods(url, target_application=TvbProfile.current.web.OPENSHIFT_APPLICATION):
        if not TvbProfile.current.web.OPENSHIFT_DEPLOY:
            return

        LOGGER.info("Notify all pods with url {}".format(url))
        openshift_pods, auth_header = KubeNotifier.get_pods(target_application)
        url_pattern = "http://{}:" + str(TvbProfile.current.web.SERVER_PORT) + url
        with ThreadPoolExecutor(max_workers=len(openshift_pods)) as executor:
            for pod in openshift_pods:
                pod_ip = pod['ip']
                LOGGER.info("Notify pod: {}".format(pod_ip))
                executor.submit(requests.get, url=url_pattern.format(pod_ip), headers=auth_header)

    @staticmethod
    def fetch_endpoints(auth_header, target_application=TvbProfile.current.web.OPENSHIFT_APPLICATION):
        response = requests.get(
            url='https://kubernetes.default.svc/api/v1/namespaces/{}/endpoints/{}'.format(
                TvbProfile.current.web.OPENSHIFT_NAMESPACE, target_application),
            verify=False, headers=auth_header)
        response.raise_for_status()
        return response
