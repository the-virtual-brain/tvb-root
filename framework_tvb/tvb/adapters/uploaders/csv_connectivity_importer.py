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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import csv
import numpy
from tvb.basic.logger.builder import get_logger
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.core.neotraits.forms import TraitUploadField, SelectField, TraitDataTypeSelectField
from tvb.core.neocom import h5


class CSVConnectivityParser(object):
    """
    Parser for a connectivity csv file
    Such a file may begin with a optional header of ordinal integers
    The body of the file is a square matrix of floats
    -1 is interpreted as 0
    If a header is present the matrices columns and rows are permuted
    so that the header ordinals would be in ascending order
    """

    def __init__(self, csv_file, delimiter=','):
        self.rows = list(csv.reader(csv_file, delimiter=str(delimiter)))
        self.connectivity_size = len(self.rows[0])
        self.line = 0
        self.permutation = list(range(self.connectivity_size))
        """ A permutation represented as a list index -> new_index. Defaults to the identity permutation"""
        self.result_conn = [[] for _ in range(self.connectivity_size)]

        rows_count = len(self.rows)
        if rows_count == self.connectivity_size + 1:
            self._parse_header()
        elif rows_count == self.connectivity_size:
            pass  # we have no header
        else:
            raise LaunchException(
                "Could not parse a number matrix. Check field delimiter. Found %d rows and %d columns" %
                (rows_count, self.connectivity_size))
        self._parse_body()

    def _parse_header(self):
        """
        Reads the ordinals from the header and updates self.permutation
        """
        self.line += 1
        try:
            ordinals = [int(v) for v in self.rows[0]]
        except ValueError:
            raise LaunchException("Invalid ordinal in header %s" % self.rows[0])

        header_i = list(enumerate(ordinals))
        header_i.sort(key=lambda i__ordinal: i__ordinal[1])  # sort by the column ordinal
        inverse_permutation = [i for i, ordinal_ in header_i]

        for i in range(len(self.permutation)):
            self.permutation[inverse_permutation[i]] = i

        self.rows = self.rows[1:]  # consume header

    def _parse_body(self):
        for row_idx, row in enumerate(self.rows):
            self.line += 1
            if len(row) != self.connectivity_size:
                msg = "Invalid Connectivity Row size! %d != %d at row %d" % (
                    len(row), self.connectivity_size, self.line)
                raise LaunchException(msg)

            new_row = [0] * self.connectivity_size

            for col_idx, col in enumerate(row):
                new_row[self.permutation[col_idx]] = max(float(col), 0)

            self.result_conn[self.permutation[row_idx]] = new_row


DELIMITER_OPTIONS = {'comma': ',', 'semicolon': ';', 'tab': '\t', 'space': ' ', 'colon': ':'}


class CSVConnectivityImporterModel(UploaderViewModel):
    weights = Str(
        label='Weights file (csv)'
    )

    weights_delimiter = Str(
        choices=tuple(DELIMITER_OPTIONS.values()),
        default=tuple(DELIMITER_OPTIONS.values())[0],
        label='Field delimiter : '
    )

    tracts = Str(
        label='Tracts file (csv)'
    )

    tracts_delimiter = Str(
        choices=tuple(DELIMITER_OPTIONS.values()),
        default=tuple(DELIMITER_OPTIONS.values())[0],
        label='Field delimiter : '
    )

    input_data = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Reference Connectivity Matrix (for node labels, 3d positions etc.)'
    )


class CSVConnectivityImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(CSVConnectivityImporterForm, self).__init__(project_id)

        self.weights = TraitUploadField(CSVConnectivityImporterModel.weights, '.csv', self.project_id,
                                        'weights', self.temporary_files)
        self.weights_delimiter = SelectField(CSVConnectivityImporterModel.weights_delimiter, self.project_id,
                                             name='weights_delimiter', choices=DELIMITER_OPTIONS)
        self.tracts = TraitUploadField(CSVConnectivityImporterModel.tracts, ['.csv'], self.project_id,
                                       'tracts', self.temporary_files)
        self.tracts_delimiter = SelectField(CSVConnectivityImporterModel.tracts_delimiter, self.project_id,
                                            name='tracts_delimiter', choices=DELIMITER_OPTIONS)
        self.input_data = TraitDataTypeSelectField(CSVConnectivityImporterModel.input_data, self.project_id,
                                                   'input_data')

    @staticmethod
    def get_view_model():
        return CSVConnectivityImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'weights': '.csv',
            'tracts': '.csv'
        }


class CSVConnectivityImporter(ABCUploader):
    """
    Handler for uploading a Connectivity csv from the dti pipeline
    """
    _ui_name = "Connectivity CSV"
    _ui_subsection = "csv_connectivity_importer"
    _ui_description = "Import a Connectivity from two CSV files as result from the DTI pipeline"
    WEIGHTS_FILE = "weights.txt"
    TRACT_FILE = "tract_lengths.txt"

    def __init__(self):
        ABCUploader.__init__(self)
        self.logger = get_logger(self.__class__.__module__)

    def get_form_class(self):
        return CSVConnectivityImporterForm

    def get_output(self):
        return [ConnectivityIndex]

    def _read_csv_file(self, csv_file, delimiter):
        """
        Read a CSV file, arrange rows/columns in the correct order,
        to obtain Weight/Tract data in TVB compatible format.
        """
        with open(csv_file, 'rU') as f:
            result_conn = CSVConnectivityParser(f, delimiter).result_conn
            self.logger.debug("Read Connectivity file of size %d" % len(result_conn))
            return numpy.array(result_conn)

    def launch(self, view_model):
        # type: (CSVConnectivityImporterModel) -> ConnectivityIndex
        """
        Execute import operations: process the weights and tracts csv files, then use
        the reference connectivity passed as input_data for the rest of the attributes.

        :raises LaunchException: when the number of nodes in CSV files doesn't match the one in the connectivity
        """
        weights_matrix = self._read_csv_file(view_model.weights, view_model.weights_delimiter)
        tract_matrix = self._read_csv_file(view_model.tracts, view_model.tracts_delimiter)
        FilesHelper.remove_files([view_model.weights, view_model.tracts])
        input_data = self.load_entity_by_gid(view_model.input_data)
        if weights_matrix.shape[0] != input_data.number_of_regions:
            raise LaunchException("The csv files define %s nodes but the connectivity you selected as reference "
                                  "has only %s nodes." % (weights_matrix.shape[0], input_data.number_of_regions))

        input_connectivity = h5.load_from_index(input_data)

        result = Connectivity()
        result.centres = input_connectivity.centres
        result.region_labels = input_connectivity.region_labels
        result.weights = weights_matrix
        result.tract_lengths = tract_matrix
        result.orientations = input_connectivity.orientations
        result.areas = input_connectivity.areas
        result.cortical = input_connectivity.cortical
        result.hemispheres = input_connectivity.hemispheres
        result.configure()

        return h5.store_complete(result, self.storage_path)
