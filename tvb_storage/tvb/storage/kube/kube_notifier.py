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
Service layer used for Kubernetes calls.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import requests
from kubernetes.config import incluster_config
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile

LOGGER = get_logger(__name__)


class KubeNotifier(object):

    @staticmethod
    def do_rest_call_to_pod(rest_app_server, rest_method_name, submit_param,
                            auth_header=None, data=None):

        LOGGER.info("Notifying POD {} method {} for param {} and data {}".format(rest_app_server, rest_method_name,
                                                                                 submit_param, data))
        if auth_header is None:
            auth_header = KubeNotifier.get_authorization_header()

        protocol = 'https' if TvbProfile.current.web.IS_CLOUD_HTTPS else 'http'
        url_pattern = "{}://{}:{}/kube/{}/{}"
        url_filled = url_pattern.format(protocol, rest_app_server, TvbProfile.current.web.SERVER_PORT,
                                        rest_method_name, submit_param)
        if data is None:
            return requests.get(url=url_filled, headers=auth_header)

        return requests.post(url=url_filled, headers=auth_header, data=data)

    @staticmethod
    def _get_authorization_token():
        kube_config = incluster_config.InClusterConfigLoader(
            token_filename=incluster_config.SERVICE_TOKEN_FILENAME,
            cert_filename=incluster_config.SERVICE_CERT_FILENAME,
            try_refresh_token=True)
        kube_config.load_and_set(None)
        return kube_config.token

    @staticmethod
    def get_authorization_header():
        token = KubeNotifier._get_authorization_token()
        return {"Authorization": "{}".format(token)}

    @staticmethod
    def check_token(authorization_token):
        expected_token = KubeNotifier._get_authorization_token()
        assert authorization_token == expected_token
