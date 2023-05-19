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
.. moduleauthor:: David Bacter <david.bacter@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import os
import json
import numpy

from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import TraitUploadField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.core.adapters.abcuploader import ABCUploaderForm, ABCUploader
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.time_series import TimeSeriesRegion


class BIDSImporterModel(UploaderViewModel):
    uploaded = Str(
        label='BIDS derivatives dataset (zip)',
        doc="data compatible with BIDS Extension Proposal 032 (BEP032): BIDS Computational Model Specification"
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
    _ui_name = "BIDS Derivatives Importer"
    _ui_subsection = "bids_importer"
    _ui_description = "Import a dataset in BIDS format"

    SUBJECT_PREFIX = "sub"

    NET_TOKEN = "net"
    COORDS_TOKEN = "coord"
    SPATIAL_TOKEN = "spatial"
    TS_TOKEN = "ts"

    TSV_EXTENSION = ".tsv"
    JSON_EXTENSION = ".json"

    WEIGHTS_FILE = "weights" + TSV_EXTENSION
    WEIGHTS_JSON_FILE = "weights" + JSON_EXTENSION
    DISTANCES_FILE = "distances" + TSV_EXTENSION

    VERTICES_FILE = "vertices" + TSV_EXTENSION
    NORMALS_FILE = "normals" + TSV_EXTENSION
    TRIANGLES_FILE = "faces" + TSV_EXTENSION

    COORDS_ROWS_KEY = "CoordsRows"

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
        subject_folders = set()

        # First we find subject parent folders
        for file_name in files:
            if self.__is_subject_folder(file_name):
                subject_folders.add(file_name)

        if len(subject_folders) == 0:
            # Try to determine subject folders in a different manner
            for file_name in files:
                possible_subject_folder = os.path.dirname(os.path.dirname(file_name))
                if self.__is_subject_folder(possible_subject_folder):
                    subject_folders.add(possible_subject_folder)

        connectivity = None
        ts_dict = None

        for subject_folder in subject_folders:
            net_folder = os.path.join(subject_folder, self.NET_TOKEN)
            if os.path.exists(net_folder):
                connectivity = self.__build_connectivity(net_folder)

            coords_folder = os.path.join(subject_folder, self.COORDS_TOKEN)
            if os.path.exists(coords_folder):
                self.__build_surface(coords_folder)

            ts_folder = os.path.join(subject_folder, self.TS_TOKEN)
            if os.path.exists(ts_folder):
                ts_dict = self.__build_time_series(ts_folder, connectivity)

            spatial_folder = os.path.join(subject_folder, self.SPATIAL_TOKEN)
            if os.path.exists(spatial_folder):
                self.__build_functional_connectivity(spatial_folder, ts_dict)

    def __is_subject_folder(self, file_name):
        return os.path.basename(file_name).startswith(self.SUBJECT_PREFIX) and os.path.isdir(file_name)

    def __build_connectivity(self, net_folder):
        weights_matrix = None
        tracts_matrix = None
        centres = None
        labels_vector = None

        for net_file_name in os.listdir(net_folder):
            net_file_path = os.path.join(net_folder, net_file_name)

            if net_file_name.endswith(self.WEIGHTS_FILE):
                weights_matrix = self.read_list_data(net_file_path)
            elif net_file_name.endswith(self.DISTANCES_FILE):
                tracts_matrix = self.read_list_data(net_file_path)
            elif net_file_name.endswith(self.WEIGHTS_JSON_FILE):
                with open(net_file_path) as json_file:
                    json_dict = json.load(json_file)
                    labels_path = json_dict[self.COORDS_ROWS_KEY][0]
                    centres_path = json_dict[self.COORDS_ROWS_KEY][1]

                    dir_path = os.path.dirname(net_file_path)
                    labels_path = os.path.join(dir_path, labels_path).replace(self.JSON_EXTENSION, self.TSV_EXTENSION)
                    centres_path = os.path.join(dir_path, centres_path).replace(self.JSON_EXTENSION, self.TSV_EXTENSION)

                    centres = self.read_list_data(centres_path)
                    labels_vector = self.read_list_data(labels_path, dtype=numpy.str_, usecols=[0])

        connectivity = Connectivity()

        expected_number_of_nodes = len(centres)
        connectivity.set_centres(centres, expected_number_of_nodes)
        connectivity.set_region_labels(labels_vector)
        connectivity.set_weights(weights_matrix, expected_number_of_nodes)
        connectivity.set_tract_lengths(tracts_matrix, expected_number_of_nodes)
        connectivity.configure()

        connectivity_index = self.store_complete(connectivity)
        self._capture_operation_results([connectivity_index])
        dao.store_entity(connectivity_index)

        return connectivity

    def __build_surface(self, surface_folder):
        vertices = None
        normals = None
        triangles = None

        for surface_file_name in os.listdir(surface_folder):
            surface_file_path = os.path.join(surface_folder, surface_file_name)

            if surface_file_name.endswith(self.VERTICES_FILE):
                vertices = self.read_list_data(surface_file_path)
            elif surface_file_name.endswith(self.NORMALS_FILE):
                normals = self.read_list_data(surface_file_path)
            elif surface_file_name.endswith(self.TRIANGLES_FILE):
                triangles = self.read_list_data(surface_file_path, dtype=numpy.int64)

        surface = CorticalSurface()
        surface.set_scaled_vertices(vertices)
        surface.normals = normals
        surface.zero_based_triangles = False
        surface.triangles = triangles - 1
        surface.hemisphere_mask = numpy.array([False] * len(vertices))
        surface.compute_triangle_normals()
        surface.valid_for_simulations = True

        validation_result = surface.validate()
        if validation_result.warnings:
            self.add_operation_additional_info(validation_result.summary())

        surface.configure()

        surface_index = self.store_complete(surface)

        self._capture_operation_results([surface_index])
        dao.store_entity(surface_index)

        return surface

    def __build_time_series(self, ts_folder, connectivity):
        tsv_ts_files = filter(lambda x: x.endswith(self.TSV_EXTENSION), os.listdir(ts_folder))
        ts_dict = {}
        for tsv_ts_file_name in tsv_ts_files:
            tsv_ts_file = os.path.join(ts_folder, tsv_ts_file_name)
            ts_array_data = self.read_list_data(tsv_ts_file)
            ts_array_data = ts_array_data.reshape((len(ts_array_data), 1, len(ts_array_data[0]), 1))

            json_ts_file = tsv_ts_file.replace(self.TSV_EXTENSION, self.JSON_EXTENSION)
            with open(json_ts_file) as json_ts:
                ts_time_file = json.load(json_ts)[self.COORDS_ROWS_KEY][0]

            dir_path = os.path.dirname(json_ts_file)
            ts_time_file = os.path.join(dir_path, ts_time_file).replace(self.JSON_EXTENSION, self.TSV_EXTENSION)
            ts_times_data = self.read_list_data(ts_time_file)

            ts = TimeSeriesRegion()
            ts.data = ts_array_data
            ts.time = ts_times_data
            ts.connectivity = connectivity

            self.generic_attributes.user_tag_1 = tsv_ts_file_name
            ts.configure()
            ts_index = self.store_complete(ts, self.generic_attributes)
            ts_index.fixed_generic_attributes = True
            self._capture_operation_results([ts_index])
            dao.store_entity(ts_index)
            ts_dict[os.path.basename(tsv_ts_file)] = ts

        return ts_dict

    def __build_functional_connectivity(self, spatial_folder, ts_dict):
        tsv_spatial_files = filter(lambda x: x.endswith(self.TSV_EXTENSION), os.listdir(spatial_folder))

        for tsv_spatial_file_name in tsv_spatial_files:
            tsv_spatial_file = os.path.join(spatial_folder, tsv_spatial_file_name)
            fc_data = self.read_list_data(tsv_spatial_file)
            fc_data = fc_data.reshape((fc_data.shape[0], fc_data.shape[1], 1, 1))

            pearson_correlation = CorrelationCoefficients()
            pearson_correlation.array_data = fc_data

            name_key = tsv_spatial_file_name.replace('sim_fc', '').replace('emp_fc', '').replace('.tsv', '')
            for ts_file_name, ts in ts_dict.items():
                if name_key in ts_file_name:
                    pearson_correlation.source = ts
                    break

            self.generic_attributes.user_tag_1 = tsv_spatial_file_name
            pearson_correlation.configure()
            pearson_correlation_index = self.store_complete(pearson_correlation, self.generic_attributes)
            pearson_correlation_index.fixed_generic_attributes = True
            self._capture_operation_results([pearson_correlation_index])
            dao.store_entity(pearson_correlation_index)
