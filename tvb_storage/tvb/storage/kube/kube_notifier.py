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
Service layer used for kubernetes calls.

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import requests
from kubernetes import config, client
from kubernetes.config import incluster_config
from concurrent.futures.thread import ThreadPoolExecutor
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile

LOGGER = get_logger(__name__)


class KubeNotifier(object):

    @staticmethod
    def get_pods(application):
        openshift_pods = None

        try:
            response = KubeNotifier.fetch_endpoints(application)
            openshift_pods = response[0].subsets[0].addresses

        except Exception as e:
            LOGGER.error("Failed to retrieve openshift pods for application {}".format(application), e)

        return openshift_pods

    @staticmethod
    def notify_pods(url, target_application=TvbProfile.current.web.OPENSHIFT_APPLICATION):

        if not TvbProfile.current.web.OPENSHIFT_DEPLOY:
            return

        LOGGER.info("Notify all pods with url {}".format(url))
        openshift_pods = KubeNotifier.get_pods(target_application)
        url_pattern = "http://{}:" + str(TvbProfile.current.web.SERVER_PORT) + url
        auth_header = KubeNotifier.get_authorization_header()

        with ThreadPoolExecutor(max_workers=len(openshift_pods)) as executor:
            for pod in openshift_pods:
                pod_ip = pod.ip
                LOGGER.info("Notify pod: {}".format(pod_ip))
                executor.submit(requests.get, url=url_pattern.format(pod_ip), headers=auth_header)

    @staticmethod
    def fetch_endpoints(target_application=TvbProfile.current.web.OPENSHIFT_APPLICATION):
        config.load_incluster_config()

        v1 = client.CoreV1Api()
        response = v1.read_namespaced_endpoints_with_http_info(target_application,
                                                               TvbProfile.current.web.OPENSHIFT_NAMESPACE)
        LOGGER.info(f"This is the response from KubeClient: {response}")
        return response

    @staticmethod
    def get_authorization_token():
        kube_config = incluster_config.InClusterConfigLoader(
            token_filename=incluster_config.SERVICE_TOKEN_FILENAME,
            cert_filename=incluster_config.SERVICE_CERT_FILENAME,
            try_refresh_token=True)
        kube_config.load_and_set(None)
        return kube_config.token

    @staticmethod
    def get_authorization_header():
        token = KubeNotifier.get_authorization_token()
        return {"Authorization": "{}".format(token)}

    @staticmethod
    def check_token(authorization_token):
        expected_token = KubeNotifier.get_authorization_token()
        assert authorization_token == expected_token
