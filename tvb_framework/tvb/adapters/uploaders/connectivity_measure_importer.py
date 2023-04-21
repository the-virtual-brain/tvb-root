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

from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField, StrField, TraitDataTypeSelectField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure


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

    def __init__(self):
        super(ConnectivityMeasureImporterForm, self).__init__()

        self.data_file = TraitUploadField(ConnectivityMeasureImporterModel.data_file, '.mat', 'data_file')
        self.dataset_name = StrField(ConnectivityMeasureImporterModel.dataset_name, 'dataset_name')
        self.connectivity = TraitDataTypeSelectField(ConnectivityMeasureImporterModel.connectivity, name='connectivity')

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
            connectivity = self.load_traited_by_gid(view_model.connectivity)

            if node_count != connectivity.number_of_regions:
                raise LaunchException('The measurements are for %s nodes but the selected connectivity'
                                      ' contains %s nodes' % (node_count, connectivity.number_of_regions))

            measures = []
            self.generic_attributes.user_tag_2 = "conn_%d" % node_count

            for i in range(measurement_count):
                cm_data = data[i, :]

                measure = ConnectivityMeasure()
                measure.array_data = cm_data
                measure.connectivity = connectivity
                measure.title = "Measure %d for Connectivity with %d nodes." % ((i + 1), node_count)

                cm_idx = self.store_complete(measure)
                measures.append(cm_idx)
            return measures

        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)
