# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import networkx
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.networkx_connectivity.parser import NetworkxParser
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.datatypes.connectivity import Connectivity


class NetworkxConnectivityImporter(ABCUploader):
    """
    Import connectivity data stored in the networkx gpickle format
    """
    _ui_name = "Connectivity Networkx gpickle"
    _ui_subsection = "networkx_importer"
    _ui_description = "Import connectivity data stored in the networkx gpickle format"


    def get_upload_input_tree(self):

        tree = [{'name': 'data_file', 'type': 'upload', 'required_type': '.gpickle',
                 'label': 'Please select file to import', 'required': True}]

        tree.extend(NetworkxParser.prepare_input_params_tree())
        return tree
        
        
    def get_output(self):
        return [Connectivity]


    @transactional
    def launch(self, data_file, **kwargs):
        try:
            parser = NetworkxParser(self.storage_path, **kwargs)
            net = networkx.read_gpickle(data_file)
            connectivity = parser.parse(net)
            return [connectivity]
        except ParseException as excep:
            self.log.exception("Could not process Connectivity")
            raise LaunchException(excep)

