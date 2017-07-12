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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import numpy
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.connectivity import Connectivity

NORMALIZATION_OPTIONS = [
    {'name': 'None', 'value': 'none'},
    {'name': 'Region (node)', 'value': 'region'},
    {'name': 'Absolute (max weight)', 'value': 'tract'}]


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


    def get_upload_input_tree(self):
        """
        Take as input a ZIP archive.
        """
        return [{'name': 'uploaded', 'type': 'upload', 'required_type': 'application/zip',
                 'label': 'Connectivity file (zip)', 'required': True},

                {'name': 'normalization', 'label': 'Weights Normalization', 'type': 'select', 'default': 'none',
                 'options': NORMALIZATION_OPTIONS, 'description': 'Normalization mode for weights'}]


    def get_output(self):
        return [Connectivity]


    def launch(self, uploaded, normalization=None):
        """
        Execute import operations: unpack ZIP and build Connectivity object as result.

        :param uploaded: an archive containing the Connectivity data to be imported

        :returns: `Connectivity`

        :raises LaunchException: when `uploaded` is empty or nonexistent
        :raises Exception: when
                    * weights or tracts matrix is invalid (negative values, wrong shape)
                    * any of the vector orientation, areas, cortical or hemisphere is \
                      different from the expected number of nodes
        """
        if uploaded is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        files = FilesHelper().unpack_zip(uploaded, self.storage_path)

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

        ### Clean remaining text-files.
        FilesHelper.remove_files(files, True)

        result = Connectivity()
        result.storage_path = self.storage_path

        ### Fill positions
        if centres is None:
            raise Exception("Region centres are required for Connectivity Regions! "
                            "We expect a file that contains *centres* inside the uploaded ZIP.")
        expected_number_of_nodes = len(centres)
        if expected_number_of_nodes < 2:
            raise Exception("A connectivity with at least 2 nodes is expected")
        result.centres = centres
        if labels_vector is not None:
            result.region_labels = labels_vector

        ### Fill and check weights
        if weights_matrix is not None:
            if weights_matrix.shape != (expected_number_of_nodes, expected_number_of_nodes):
                raise Exception("Unexpected shape for weights matrix! "
                                "Should be %d x %d " % (expected_number_of_nodes, expected_number_of_nodes))
            result.weights = weights_matrix
            if normalization:
                result.weights = result.scaled_weights(normalization)

        ### Fill and check tracts    
        if tract_matrix is not None:
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
        return result
