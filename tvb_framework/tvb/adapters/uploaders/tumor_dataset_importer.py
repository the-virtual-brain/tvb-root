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
Import Brain Tumor dataset

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

import csv
import os
import uuid
import numpy as np
import json
from tvb.adapters.datatypes.db.graph import CorrelationCoefficientsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.graph_h5 import CorrelationCoefficientsH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.adapters.uploaders.csv_connectivity_importer import CSVDelimiterOptionsEnum
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporter, ZIPConnectivityImporterModel
from tvb.config.algorithm_categories import DEFAULTDATASTATE_RAW_DATA
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField
from tvb.core.neotraits.view_model import Str, ViewModel
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.datatypes.time_series import TimeSeriesRegion, TimeSeries

WARNING_MSG = "File {} does not exist."


class TumorDatasetImporterModel(ViewModel):
    data_file = Str(
        label='Tumor Dataset (BIDS + zip)'
    )


class TumorDatasetImporterForm(ABCUploaderForm):

    def __init__(self):
        super(TumorDatasetImporterForm, self).__init__()
        self.data_file = TraitUploadField(TumorDatasetImporterModel.data_file, '.zip', 'data_file')

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.zip'
        }

    @staticmethod
    def get_view_model():
        return TumorDatasetImporterModel


class TumorDatasetImporter(ABCAdapter):
    _ui_name = "Tumor Dataset"
    _ui_description = "Download manually Tumor Dataset from the EBRAINS KG and import it into TVB."

    MAXIMUM_DOWNLOAD_RETRIES = 3
    SLEEP_TIME = 3

    CONN_ZIP_FILE = "SC.zip"
    FC_MAT_FILE = "FC.mat"
    FC_DATASET_NAME = "FC_cc_DK68"
    TIME_SERIES_CSV_FILE = "HRF.csv"

    def get_form_class(self):
        return TumorDatasetImporterForm

    def get_output(self):
        return []

    def get_required_disk_size(self, view_model):
        return -1

    def get_required_memory_size(self, view_model):
        return -1

    def __import_tumor_connectivity(self, conn_folder, patient, user_tag):

        connectivity_zip = os.path.join(conn_folder, self.CONN_ZIP_FILE)
        if not os.path.exists(connectivity_zip):
            self.log.warning(WARNING_MSG.format(connectivity_zip))
            return

        import_conn_adapter = self.build_adapter_from_class(ZIPConnectivityImporter)
        operation = dao.get_operation_by_id(self.operation_id)
        import_conn_adapter.extract_operation_data(operation)
        import_conn_model = ZIPConnectivityImporterModel()
        import_conn_model.uploaded = connectivity_zip
        import_conn_model.data_subject = patient
        import_conn_model.generic_attributes.user_tag_1 = user_tag

        connectivity_index = import_conn_adapter.launch(import_conn_model)

        self.generic_attributes.subject = patient
        self.generic_attributes.user_tag_1 = user_tag
        self._capture_operation_results([connectivity_index])
        connectivity_index.fk_from_operation = self.operation_id
        dao.store_entity(connectivity_index)

        return connectivity_index.gid

    def __import_time_series_csv_datatype(self, hrf_folder, connectivity_gid, patient, user_tag):

        path = os.path.join(hrf_folder, self.TIME_SERIES_CSV_FILE)
        if not os.path.exists(path):
            self.log.warning(WARNING_MSG.format(path))
            return

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=CSVDelimiterOptionsEnum.COMMA.value)
            ts = list(csv_reader)

        ts_data = np.array(ts, dtype=np.float64).reshape((len(ts), 1, len(ts[0]), 1))
        ts_time = np.random.rand(ts_data.shape[0], )

        project = dao.get_project_by_id(self.current_project_id)

        ts_gid = uuid.uuid4()
        h5_path = "TimeSeries_{}.h5".format(ts_gid.hex)
        operation_folder = self.storage_interface.get_project_folder(project.name, str(self.operation_id))
        h5_path = os.path.join(operation_folder, h5_path)

        conn = h5.load_from_gid(connectivity_gid)
        ts = TimeSeriesRegion()
        ts.data = ts_data
        ts.time = ts_time
        ts.gid = ts_gid
        ts.connectivity = conn
        generic_attributes = GenericAttributes()
        generic_attributes.user_tag_1 = user_tag
        generic_attributes.state = DEFAULTDATASTATE_RAW_DATA

        with TimeSeriesRegionH5(h5_path) as ts_h5:
            ts_h5.store(ts)
            ts_h5.nr_dimensions.store(4)
            ts_h5.subject.store(patient)
            ts_h5.store_generic_attributes(generic_attributes)

        ts_index = TimeSeriesIndex()
        ts_index.gid = ts_gid.hex
        ts_index.fk_from_operation = self.operation_id
        ts_index.time_series_type = "TimeSeriesRegion"
        ts_index.data_length_1d = ts_data.shape[0]
        ts_index.data_length_2d = ts_data.shape[1]
        ts_index.data_length_3d = ts_data.shape[2]
        ts_index.data_length_4d = ts_data.shape[3]
        ts_index.data_ndim = len(ts_data.shape)
        ts_index.sample_period_unit = 'ms'
        ts_index.sample_period = TimeSeries.sample_period.default
        ts_index.sample_rate = 1024.0
        ts_index.subject = patient
        ts_index.state = DEFAULTDATASTATE_RAW_DATA
        ts_index.labels_ordering = json.dumps(list(TimeSeries.labels_ordering.default))
        ts_index.labels_dimensions = json.dumps(TimeSeries.labels_dimensions.default)
        ts_index.visible = False  # we don't want to show these TimeSeries because they are dummy
        dao.store_entity(ts_index)

        return ts_gid

    def __import_pearson_coefficients_datatype(self, fc_folder, patient, user_tag, ts_gid):

        path = os.path.join(fc_folder, self.FC_MAT_FILE)
        if not os.path.exists(path):
            self.log.warning(WARNING_MSG.format(path))
            return

        result = ABCUploader.read_matlab_data(path, self.FC_DATASET_NAME)
        result = result.reshape((result.shape[0], result.shape[1], 1, 1))

        project = dao.get_project_by_id(self.current_project_id)

        pearson_gid = uuid.uuid4()
        h5_path = "CorrelationCoefficients_{}.h5".format(pearson_gid.hex)
        operation_folder = self.storage_interface.get_project_folder(project.name, str(self.operation_id))
        h5_path = os.path.join(operation_folder, h5_path)

        generic_attributes = GenericAttributes()
        generic_attributes.user_tag_1 = user_tag
        generic_attributes.state = DEFAULTDATASTATE_RAW_DATA

        with CorrelationCoefficientsH5(h5_path) as pearson_correlation_h5:
            pearson_correlation_h5.array_data.store(result)
            pearson_correlation_h5.gid.store(pearson_gid)
            pearson_correlation_h5.source.store(ts_gid)
            pearson_correlation_h5.labels_ordering.store(CorrelationCoefficients.labels_ordering.default)
            pearson_correlation_h5.subject.store(patient)
            pearson_correlation_h5.store_generic_attributes(generic_attributes)

        pearson_correlation_index = CorrelationCoefficientsIndex()
        pearson_correlation_index.gid = pearson_gid.hex
        pearson_correlation_index.fk_from_operation = self.operation_id
        pearson_correlation_index.subject = patient
        pearson_correlation_index.state = DEFAULTDATASTATE_RAW_DATA
        pearson_correlation_index.ndim = 4
        pearson_correlation_index.fk_source_gid = ts_gid.hex  # we need a random gid here to store the index
        pearson_correlation_index.has_valid_time_series = False
        dao.store_entity(pearson_correlation_index)

    def __import_from_folder(self, datatype_folder, patient, user_tag):
        conn_gid = self.__import_tumor_connectivity(datatype_folder, patient, user_tag)
        # The Time Series are invisible in the UI and are imported
        # just so we can link them with the Pearson Coefficients
        ts_gid = self.__import_time_series_csv_datatype(datatype_folder, conn_gid, patient, user_tag)
        self.__import_pearson_coefficients_datatype(datatype_folder, patient, user_tag, ts_gid)

    def launch(self, view_model):
        # type: (TumorDatasetImporterModel) -> []
        """
        Download the Tumor Dataset and then import its data
        (currently only the connectivities and pearson coefficients (FC) are imported).
        """
        structure = self.storage_interface.unpack_zip(view_model.data_file, self.get_storage_path())
        subject_folders = {}
        for name in structure:
            if os.path.isdir(name):
                user_tag = os.path.split(name)[1]
                if user_tag.startswith("sub-"):
                    subject_folders[user_tag] = name

        for patient, patient_path in subject_folders.items():
            root_folder_imported = False
            for user_tag in os.listdir(patient_path):
                datatype_folder = os.path.join(patient_path, user_tag)
                if os.path.isdir(datatype_folder):
                    self.__import_from_folder(datatype_folder, patient, user_tag)
                elif not root_folder_imported:
                    root_folder_imported = True
                    self.__import_from_folder(patient_path, patient, "")

        self.log.debug("Importing Tumor Dataset has been successfully completed!")
