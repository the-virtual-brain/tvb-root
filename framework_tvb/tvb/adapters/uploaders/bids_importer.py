# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import json
import numpy

from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import TraitUploadField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.core.adapters.abcuploader import ABCUploaderForm, ABCUploader
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.datatypes.time_series import TimeSeriesRegion


class BIDSImporterModel(UploaderViewModel):
    uploaded = Str(
        label='BIDS dataset (zip)'
    )


class BIDSImporterForm(ABCUploaderForm):

    def __init__(self):
        super(BIDSImporterForm, self).__init__()

        self.uploaded = TraitUploadField(BIDSImporterModel.uploaded, '.zip', 'uploaded')

    @staticmethod
    def get_view_model():
        return BIDSImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'uploaded': '.zip'
        }


class BIDSImporter(ABCUploader):

    _ui_name = "BIDS Importer"
    _ui_subsection = "bids_importer"
    _ui_description = "Import a dataset in BIDS format"

    SUBJECT_PREFIX = "sub"

    NET_TOKEN = "net"
    SPATIAL_TOKEN = "spatial"
    TS_TOKEN = "ts"

    def get_form_class(self):
        return BIDSImporterForm

    def get_output(self):
        return [VolumeIndex, StructuralMRIIndex, TimeSeriesVolumeIndex, RegionVolumeMappingIndex]

    def launch(self, view_model):
        """
        Import a dataset in BIDS format
        """

        if view_model.uploaded is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        files = self.storage_interface.unpack_zip(view_model.uploaded, self.get_storage_path())
        subject_folders = []

        # First we find subject parent folders
        for file_name in files:
            if os.path.basename(file_name).startswith(self.SUBJECT_PREFIX) and os.path.isdir(file_name):
                subject_folders.append(file_name)

        connectivity = None
        ts = None

        for subject_folder in subject_folders:
            net_folder = os.path.join(subject_folder, self.NET_TOKEN)
            if os.path.exists(net_folder):
                weights_matrix = None
                tracts_matrix = None
                centres = None
                labels_vector = None

                for net_file_name in os.listdir(net_folder):
                    net_file_path = os.path.join(net_folder, net_file_name)

                    if net_file_name.endswith('weights.tsv'):
                        weights_matrix = self.read_list_data(net_file_path)
                    elif net_file_name.endswith('distances.tsv'):
                        tracts_matrix = self.read_list_data(net_file_path)
                    elif net_file_name.endswith('weights.json'):
                        with open(net_file_path) as json_file:
                            json_dict = json.load(json_file)
                            labels_path = json_dict['CoordsRows'][0]
                            centres_path = json_dict['CoordsRows'][1]

                            dir_path = os.path.dirname(net_file_path)
                            labels_path = os.path.join(dir_path, labels_path).replace('.json', '.tsv')
                            centres_path = os.path.join(dir_path, centres_path).replace('.json', '.tsv')

                            centres = self.read_list_data(centres_path)
                            labels_vector = self.read_list_data(labels_path, dtype=numpy.str, usecols=[0])

                connectivity = Connectivity()

                expected_number_of_nodes = ZIPConnectivityImporter.check_centres(centres)
                connectivity.centres = centres

                if labels_vector is not None:
                    connectivity.region_labels = labels_vector

                # Fill and check weights
                if weights_matrix is not None:
                    if weights_matrix.shape != (expected_number_of_nodes, expected_number_of_nodes):
                        raise Exception("Unexpected shape for weights matrix! "
                                        "Should be %d x %d " % (expected_number_of_nodes, expected_number_of_nodes))
                    connectivity.weights = weights_matrix

                ZIPConnectivityImporter.check_tracts(tracts_matrix, expected_number_of_nodes)
                connectivity.tract_lengths = tracts_matrix

                connectivity.configure()
                connectivity_index = self.store_complete(connectivity)
                self._capture_operation_results([connectivity_index])
                connectivity_index.fk_from_operation = self.operation_id
                dao.store_entity(connectivity_index)

            ts_folder = os.path.join(subject_folder, self.TS_TOKEN)
            if os.path.exists(ts_folder):
                tsv_ts_files = filter(lambda x: x.endswith('.tsv'), os.listdir(ts_folder))
                for tsv_ts_file_name in tsv_ts_files:
                    tsv_ts_file = os.path.join(ts_folder, tsv_ts_file_name)
                    ts_array_data = self.read_list_data(tsv_ts_file)
                    ts_array_data = ts_array_data.reshape((len(ts_array_data), 1, len(ts_array_data[0]), 1))

                    json_ts_file = tsv_ts_file.replace('.tsv', '.json')
                    with open(json_ts_file) as json_ts:
                        ts_time_file = json.load(json_ts)['CoordsRows'][0]

                    dir_path = os.path.dirname(json_ts_file)
                    ts_time_file = os.path.join(dir_path, ts_time_file).replace('.json', '.tsv')
                    ts_times_data = self.read_list_data(ts_time_file)

                    ts = TimeSeriesRegion()
                    ts.data = ts_array_data
                    ts.time = ts_times_data
                    ts.connectivity = connectivity

                    ts.configure()
                    ts_index = self.store_complete(ts)
                    self._capture_operation_results([ts_index])
                    ts_index.fk_from_operation = self.operation_id
                    dao.store_entity(ts_index)

            spatial_folder = os.path.join(subject_folder, self.SPATIAL_TOKEN)
            if os.path.exists(spatial_folder):
                tsv_spatial_files = filter(lambda x: x.endswith('.tsv'), os.listdir(spatial_folder))

                for tsv_spatial_file_name in tsv_spatial_files:
                    tsv_spatial_file = os.path.join(spatial_folder, tsv_spatial_file_name)
                    fc_data = self.read_list_data(tsv_spatial_file)
                    fc_data = fc_data.reshape((fc_data.shape[0], fc_data.shape[1], 1, 1))

                    pearson_correlation = CorrelationCoefficients()
                    pearson_correlation.array_data = fc_data
                    pearson_correlation.source = ts

                    pearson_correlation.configure()
                    pearson_correlation_index = self.store_complete(pearson_correlation)
                    self._capture_operation_results([pearson_correlation_index])
                    pearson_correlation_index.fk_from_operation = self.operation_id
                    dao.store_entity(pearson_correlation_index)
