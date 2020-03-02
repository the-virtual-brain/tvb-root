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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import uuid
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.adapters.datatypes.h5.graph_h5 import ConnectivityMeasureH5
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.core.entities.storage import transactional
from tvb.core.neotraits.forms import TraitUploadField, StrField, TraitDataTypeSelectField
from tvb.core.neotraits.db import from_ndarray
from tvb.core.neocom import h5
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity


class ConnectivityMeasureImporterModel(UploaderViewModel):
    data_file = Str(
        label='Connectivity measure file (.mat format)'
    )

    dataset_name = Attr(
        field_type=str,
        default='M',
        label='Matlab dataset name',
        doc='Name of the MATLAB dataset where data is stored'
    )

    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Large Scale Connectivity',
        doc='The Connectivity for which these measurements were made'
    )


class ConnectivityMeasureImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(ConnectivityMeasureImporterForm, self).__init__(prefix, project_id)

        self.data_file = TraitUploadField(ConnectivityMeasureImporterModel.data_file, '.mat', self, name='data_file')
        self.dataset_name = StrField(ConnectivityMeasureImporterModel.dataset_name, self, name='dataset_name')
        self.connectivity = TraitDataTypeSelectField(ConnectivityMeasureImporterModel.connectivity, self,
                                                     name='connectivity')

    @staticmethod
    def get_view_model():
        return ConnectivityMeasureImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.mat'
        }


class ConnectivityMeasureImporter(ABCUploader):
    """
    This imports a searies of conectivity measures from a .mat file
    """
    _ui_name = "ConnectivityMeasure"
    _ui_subsection = "connectivity_measure"
    _ui_description = "Import a searies of connectivity measures from a .mat file"

    def get_form_class(self):
        return ConnectivityMeasureImporterForm

    def get_output(self):
        return [ConnectivityMeasureIndex]

    @transactional
    def launch(self, view_model):
        # type: (ConnectivityMeasureImporterModel) -> [ConnectivityMeasureIndex]
        """
        Execute import operations:
        """
        try:
            data = self.read_matlab_data(view_model.data_file, view_model.dataset_name)
            measurement_count, node_count = data.shape
            connectivity_index = self.load_entity_by_gid(view_model.connectivity.hex)

            if node_count != connectivity_index.number_of_regions:
                raise LaunchException('The measurements are for %s nodes but the selected connectivity'
                                      ' contains %s nodes' % (node_count, connectivity_index.number_of_regions))

            measures = []
            for i in range(measurement_count):
                cm_idx = ConnectivityMeasureIndex()
                cm_idx.type = ConnectivityMeasureIndex.__name__
                cm_idx.connectivity_gid = connectivity_index.gid

                cm_data = data[i, :]
                cm_idx.array_data_ndim = cm_data.ndim
                cm_idx.ndim = cm_data.ndim
                cm_idx.array_data_min, cm_idx.array_data_max, cm_idx.array_data_mean = from_ndarray(cm_data)

                cm_h5_path = h5.path_for(self.storage_path, ConnectivityMeasureH5, cm_idx.gid)
                with ConnectivityMeasureH5(cm_h5_path) as cm_h5:
                    cm_h5.array_data.store(data[i, :])
                    cm_h5.connectivity.store(uuid.UUID(connectivity_index.gid))
                    cm_h5.gid.store(uuid.UUID(cm_idx.gid))

                cm_idx.user_tag_2 = "nr.-%d" % (i + 1)
                cm_idx.user_tag_3 = "conn_%d" % node_count
                measures.append(cm_idx)
            return measures
        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)
