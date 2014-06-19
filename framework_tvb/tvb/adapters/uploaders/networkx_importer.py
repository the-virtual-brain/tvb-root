# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
from tvb.adapters.uploaders.handler_connectivity import networkx_cmt_2connectivity
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.datatypes.connectivity import Connectivity

class NetworkxConnectivityImporter(ABCUploader):
    """
    Import connectivity data stored in the networkx gpickle format
    """
    _ui_name = "Connectivity networkx gpickle"
    _ui_subsection = "networkx_importer"
    _ui_description = "Import connectivity data stored in the networkx gpickle format"


    def get_upload_input_tree(self):
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.gpickle',
                 'label': 'Please select file to import', 'required': True}]
        
        
    def get_output(self):
        return [Connectivity]


    @transactional
    def launch(self, data_file):
        try:
            net = networkx.read_gpickle(data_file)
            conn = networkx_cmt_2connectivity(net, self.storage_path)
            return [conn]
        except ParseException, excep:
            self.log.exception(excep)
            raise LaunchException(excep)