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
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.uploaders.networkx_connectivity.parser import NetworkxParser
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.storage import transactional
from tvb.core.neotraits._forms import UploadField, SimpleStrField
from tvb.interfaces.neocom._h5loader import DirLoader


class NetworkxCFFCommonImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None, label_prefix=''):
        super(NetworkxCFFCommonImporterForm, self).__init__(prefix, project_id)
        self.key_edge_weight = SimpleStrField(self, name='key_edge_weight', default=NetworkxParser.KEY_EDGE_WEIGHT[0],
                                              label=label_prefix + 'Key Edge Weight')
        self.key_edge_tract = SimpleStrField(self, name='key_edge_tract', default=NetworkxParser.KEY_EDGE_TRACT[0],
                                             label=label_prefix + 'Key Edge Tract')
        self.key_node_coordinates = SimpleStrField(self, name='key_node_coordinates',
                                                   default=NetworkxParser.KEY_NODE_COORDINATES[0],
                                                   label=label_prefix + 'Key Node Coordinates')
        self.key_node_label = SimpleStrField(self, name='key_node_label', default=NetworkxParser.KEY_NODE_LABEL[0],
                                             label=label_prefix + 'Key Node Label')
        self.key_node_region = SimpleStrField(self, name='key_node_region', default=NetworkxParser.KEY_NODE_REGION[0],
                                              label=label_prefix + 'Key Node Region')
        self.key_node_hemisphere = SimpleStrField(self, name='key_node_hemisphere',
                                                  default=NetworkxParser.KEY_NODE_HEMISPHERE[0],
                                                  label=label_prefix + 'Key Node Hemisphere')


class NetworkxConnectivityImporterForm(NetworkxCFFCommonImporterForm):

    def __init__(self, prefix='', project_id=None):
        super(NetworkxConnectivityImporterForm, self).__init__(prefix, project_id)
        self.data_file = UploadField('.gpickle', self, name='data_file', required=True, label='Please select file to import')


class NetworkxConnectivityImporter(ABCUploader):
    """
    Import connectivity data stored in the networkx gpickle format
    """
    _ui_name = "Connectivity Networkx gpickle"
    _ui_subsection = "networkx_importer"
    _ui_description = "Import connectivity data stored in the networkx gpickle format"

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return NetworkxConnectivityImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [ConnectivityIndex]


    @transactional
    def launch(self, data_file, **kwargs):
        try:
            parser = NetworkxParser(**kwargs)
            net = networkx.read_gpickle(data_file)
            connectivity = parser.parse(net)

            conn_idx = ConnectivityIndex()
            conn_idx.fill_from_has_traits(connectivity)

            loader = DirLoader(self.storage_path)
            conn_h5_path = loader.path_for(ConnectivityH5, conn_idx.gid)

            with ConnectivityH5(conn_h5_path) as conn_h5:
                conn_h5.store(connectivity)

            return [conn_idx]
        except ParseException as excep:
            self.log.exception("Could not process Connectivity")
            raise LaunchException(excep)

