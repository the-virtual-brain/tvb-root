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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import numpy
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, Attr
from tvb.core.neotraits.forms import TraitUploadField, SelectField
from tvb.core.neocom import h5
from tvb.datatypes.connectivity import Connectivity


NORMALIZATION_OPTIONS = {'Region (node)': 'region', 'Absolute (max weight)': 'tract'}


class ZIPConnectivityImporterModel(UploaderViewModel):
    uploaded = Str(
        label='Connectivity file (zip)'
    )

    normalization = Str(
        required=False,
        choices=tuple(NORMALIZATION_OPTIONS.values()),
        label='Weights Normalization',
        doc='Normalization mode for weights'
    )


class ZIPConnectivityImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(ZIPConnectivityImporterForm, self).__init__(project_id)

        self.uploaded = TraitUploadField(ZIPConnectivityImporterModel.uploaded, '.zip', self.project_id, 'uploaded',
                                         self.temporary_files)
        self.normalization = SelectField(ZIPConnectivityImporterModel.normalization, self.project_id,
                                         name='normalization', choices=NORMALIZATION_OPTIONS)

    @staticmethod
    def get_view_model():
        return ZIPConnectivityImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'uploaded': '.zip'
        }


class ZIPConnectivityImporter(ABCUploader):
    """
    Handler for uploading a Connectivity archive, with files holding 
    text export of connectivity data from Numpy arrays.
    """
    _ui_name = "Connectivity ZIP"
    _ui_subsection = "zip_connectivity_importer"
    _ui_description = "Import a Connectivity from ZIP"

    WEIGHT_TOKEN = "weight"
    CENTRES_TOKEN = "centres"
    CENTRES_TOKEN2 = "centers"
    TRACT_TOKEN = "tract"
    ORIENTATION_TOKEN = "orientation"
    AREA_TOKEN = "area"
    CORTICAL_INFO = "cortical"
    HEMISPHERE_INFO = "hemisphere"

    def get_form_class(self):
        return ZIPConnectivityImporterForm

    def get_output(self):
        return [ConnectivityIndex]

    def launch(self, view_model):
        # type: (ZIPConnectivityImporterModel) -> [ConnectivityIndex]
        """
        Execute import operations: unpack ZIP and build Connectivity object as result.
        :raises LaunchException: when `uploaded` is empty or nonexistent
        :raises Exception: when
                    * weights or tracts matrix is invalid (negative values, wrong shape)
                    * any of the vector orientation, areas, cortical or hemisphere is \
                      different from the expected number of nodes
        """
        if view_model.uploaded is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        files = FilesHelper().unpack_zip(view_model.uploaded, self.storage_path)

        weights_matrix = None
        centres = None
        labels_vector = None
        tract_matrix = None
        orientation = None
        areas = None
        cortical_vector = None
        hemisphere_vector = None

        for file_name in files:
            file_name_low = file_name.lower()
            if self.WEIGHT_TOKEN in file_name_low:
                weights_matrix = self.read_list_data(file_name)
            elif self.CENTRES_TOKEN in file_name_low or self.CENTRES_TOKEN2 in file_name_low:
                centres = self.read_list_data(file_name, usecols=[1, 2, 3])
                labels_vector = self.read_list_data(file_name, dtype=numpy.str, usecols=[0])
            elif self.TRACT_TOKEN in file_name_low:
                tract_matrix = self.read_list_data(file_name)
            elif self.ORIENTATION_TOKEN in file_name_low:
                orientation = self.read_list_data(file_name)
            elif self.AREA_TOKEN in file_name_low:
                areas = self.read_list_data(file_name)
            elif self.CORTICAL_INFO in file_name_low:
                cortical_vector = self.read_list_data(file_name, dtype=numpy.bool)
            elif self.HEMISPHERE_INFO in file_name_low:
                hemisphere_vector = self.read_list_data(file_name, dtype=numpy.bool)

        # Clean remaining text-files.
        FilesHelper.remove_files(files, True)

        result = Connectivity()

        # Fill positions
        if centres is None:
            raise Exception("Region centres are required for Connectivity Regions! "
                            "We expect a file that contains *centres* inside the uploaded ZIP.")
        expected_number_of_nodes = len(centres)
        if expected_number_of_nodes < 2:
            raise Exception("A connectivity with at least 2 nodes is expected")
        result.centres = centres
        if labels_vector is not None:
            result.region_labels = labels_vector

        # Fill and check weights
        if weights_matrix is not None:
            if weights_matrix.shape != (expected_number_of_nodes, expected_number_of_nodes):
                raise Exception("Unexpected shape for weights matrix! "
                                "Should be %d x %d " % (expected_number_of_nodes, expected_number_of_nodes))
            result.weights = weights_matrix
            if view_model.normalization:
                result.weights = result.scaled_weights(view_model.normalization)

        # Fill and check tracts. Allow empty files for tracts, they will be computed by tvb-library.
        if tract_matrix is not None:
            if tract_matrix.size != 0:
                if numpy.any([x < 0 for x in tract_matrix.flatten()]):
                    raise Exception("Negative values are not accepted in tracts matrix! "
                                    "Please check your file, and use values >= 0")
                if tract_matrix.shape != (expected_number_of_nodes, expected_number_of_nodes):
                    raise Exception("Unexpected shape for tracts matrix! "
                                    "Should be %d x %d " % (expected_number_of_nodes, expected_number_of_nodes))
            result.tract_lengths = tract_matrix

        if orientation is not None:
            if len(orientation) != expected_number_of_nodes:
                raise Exception("Invalid size for vector orientation. "
                                "Expected the same as region-centers number %d" % expected_number_of_nodes)
            result.orientations = orientation

        if areas is not None:
            if len(areas) != expected_number_of_nodes:
                raise Exception("Invalid size for vector areas. "
                                "Expected the same as region-centers number %d" % expected_number_of_nodes)
            result.areas = areas

        if cortical_vector is not None:
            if len(cortical_vector) != expected_number_of_nodes:
                raise Exception("Invalid size for vector cortical. "
                                "Expected the same as region-centers number %d" % expected_number_of_nodes)
            result.cortical = cortical_vector

        if hemisphere_vector is not None:
            if len(hemisphere_vector) != expected_number_of_nodes:
                raise Exception("Invalid size for vector hemispheres. "
                                "Expected the same as region-centers number %d" % expected_number_of_nodes)
            result.hemispheres = hemisphere_vector

        result.configure()
        return h5.store_complete(result, self.storage_path)
