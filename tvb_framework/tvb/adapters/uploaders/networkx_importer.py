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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import pandas
from tvb.adapters.uploaders.networkx_connectivity.parser import NetworkxParser
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.entities.storage import transactional
from tvb.core.neotraits.forms import TraitUploadField, StrField
from tvb.core.neocom import h5
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str


class NetworkxImporterModel(UploaderViewModel):
    data_file = Str(
        label='Please select file to import'
    )

    key_edge_weight = Attr(
        field_type=str,
        required=False,
        default=NetworkxParser.KEY_EDGE_WEIGHT[0],
        label='Key Edge Weight'
    )

    key_edge_tract = Attr(
        field_type=str,
        required=False,
        default=NetworkxParser.KEY_EDGE_TRACT[0],
        label='Key Edge Tract'
    )

    key_node_coordinates = Attr(
        field_type=str,
        required=False,
        default=NetworkxParser.KEY_NODE_COORDINATES[0],
        label='Key Node Coordinates'
    )

    key_node_label = Attr(
        field_type=str,
        required=False,
        default=NetworkxParser.KEY_NODE_LABEL[0],
        label='Key Node Label'
    )

    key_node_region = Attr(
        field_type=str,
        required=False,
        default=NetworkxParser.KEY_NODE_REGION[0],
        label='Key Node Region'
    )

    key_node_hemisphere = Attr(
        field_type=str,
        required=False,
        default=NetworkxParser.KEY_NODE_HEMISPHERE[0],
        label='Key Node Hemisphere'
    )


class NetworkxConnectivityImporterForm(ABCUploaderForm):

    def __init__(self):
        super(NetworkxConnectivityImporterForm, self).__init__()
        self.data_file = TraitUploadField(NetworkxImporterModel.data_file, '.gpickle', 'data_file')
        self.key_edge_weight = StrField(NetworkxImporterModel.key_edge_weight, 'key_edge_weight')
        self.key_edge_tract = StrField(NetworkxImporterModel.key_edge_tract, name='key_edge_tract')
        self.key_node_coordinates = StrField(NetworkxImporterModel.key_node_coordinates, name='key_node_coordinates')
        self.key_node_label = StrField(NetworkxImporterModel.key_node_label, name='key_node_label')
        self.key_node_region = StrField(NetworkxImporterModel.key_node_region, name='key_node_region')
        self.key_node_hemisphere = StrField(NetworkxImporterModel.key_node_hemisphere, name='key_node_hemisphere')

    @staticmethod
    def get_view_model():
        return NetworkxImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.gpickle'
        }


class NetworkxConnectivityImporter(ABCUploader):
    """
    Import connectivity data stored in the networkx gpickle format
    """
    _ui_name = "Connectivity Networkx gpickle"
    _ui_subsection = "networkx_importer"
    _ui_description = "Import connectivity data stored in the networkx gpickle format"

    def get_form_class(self):
        return NetworkxConnectivityImporterForm

    def get_output(self):
        return [ConnectivityIndex]

    @transactional
    def launch(self, view_model):
        # type: (NetworkxImporterModel) -> [ConnectivityIndex]
        try:
            parser = NetworkxParser(view_model)
            net = pandas.read_pickle(view_model.data_file)
            connectivity = parser.parse(net)
            return self.store_complete(connectivity)
        except ParseException as excep:
            self.log.exception("Could not process Connectivity")
            raise LaunchException(excep)
