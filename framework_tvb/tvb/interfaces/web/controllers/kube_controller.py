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

'''
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
'''
import cherrypy
from tvb.core.services.backend_clients.standalone_client import StandAloneClient, LOCKS_QUEUE
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import check_kube_user


class KubeController(BaseController):
    @cherrypy.expose
    @check_kube_user
    def stop_operation_process(self, operation_id):
        self.logger.info("Received a request to stop process for operation {}".format(operation_id))
        StandAloneClient.stop_operation_process(int(operation_id))

    @cherrypy.expose
    @check_kube_user
    def start_operation_pod(self, operation_id):
        self.logger.info("Received a request to start operation {}".format(operation_id))
        if LOCKS_QUEUE.qsize() == 0:
            self.logger.info("Cannot start operation {} because queue is full.".format(operation_id))
            return
        LOCKS_QUEUE.get()
        StandAloneClient.start_operation(operation_id)
