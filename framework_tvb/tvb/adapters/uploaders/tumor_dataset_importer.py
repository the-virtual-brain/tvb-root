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
Import Brain Tumor dataset

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""
import csv
import os
import uuid
import requests
import time
from pathlib import Path
import numpy as np
import json

from tvb.adapters.datatypes.db.graph import CorrelationCoefficientsIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.graph_h5 import CorrelationCoefficientsH5
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.adapters.uploaders.csv_connectivity_importer import DELIMITER_OPTIONS
from tvb.adapters.uploaders.zip_connectivity_importer import ZIPConnectivityImporter, ZIPConnectivityImporterModel
from tvb.basic.logger.builder import get_logger
from tvb.config.algorithm_categories import DEFAULTDATASTATE_RAW_DATA
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcuploader import ABCUploader
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.basic.neotraits.api import Final
from tvb.datatypes.graph import CorrelationCoefficients
from tvb.datatypes.time_series import TimeSeriesRegion, TimeSeries
from tvb.storage.storage_interface import StorageInterface


class TumorDatasetImporterModel(UploaderViewModel):
    tumor_dataset_url = Final(
        label='URL for downloading the Tumor Dataset',
        default='https://kg.ebrains.eu/proxy/export?container=https://object.cscs.ch/v1/AUTH_6ebec77683fb4' \
                '72f94d352be92b5a577/hbp-d000001_TVB_brain_tumor_pub'
    )


class TumorDatasetImporterForm(ABCAdapterForm):

    @staticmethod
    def get_required_datatype():
        pass

    @staticmethod
    def get_filters():
        pass

    @staticmethod
    def get_input_name():
        return None

    @staticmethod
    def get_view_model():
        return TumorDatasetImporterModel


class TumorDatasetImporter(ABCUploader):
    _ui_name = "Import Tumor Dataset"
    _ui_description = "Download Tumor Dataset from the web and import it into TVB."

    MAXIMUM_DOWNLOAD_RETRIES = 3
    SLEEP_TIME = 3

    CONN_ZIP_FILE = "SC.zip"
    FC_MAT_FILE = "FC.mat"
    FC_DATASET_NAME = "FC_cc_DK68"
    TIME_SERIES_CSV_FILE = "HRF.csv"
    TUMOR_ZIP_FILE_NAME = 'tumor_brain_dataset.zip'

    logger = get_logger(__name__)

    def get_form_class(self):
        return TumorDatasetImporterForm

    def get_output(self):
        return []

    def _prelaunch(self, operation, view_model, available_disk_space=0):
        """
        Overwrite method in order to return the correct number of stored datatypes.
        """
        self.nr_of_datatypes = 0
        msg, _ = ABCUploader._prelaunch(self, operation, view_model, available_disk_space)
        return msg, self.nr_of_datatypes

    def __import_tumor_connectivity(self, conn_folder, patient, user_tag, storage_path):
        connectivity_zip = os.path.join(conn_folder, self.CONN_ZIP_FILE)
        if not os.path.exists(connectivity_zip):
            self.logger.error("File {} does not exist.".format(connectivity_zip))
            return
        import_conn_adapter = self.build_adapter_from_class(ZIPConnectivityImporter)
        import_conn_model = ZIPConnectivityImporterModel()
        import_conn_model.uploaded = connectivity_zip
        import_conn_model.data_subject = patient
        import_conn_model.generic_attributes.user_tag_1 = user_tag

        import_conn_adapter.storage_path = storage_path
        connectivity_index = import_conn_adapter.launch(import_conn_model)

        self.generic_attributes.subject = patient
        self.generic_attributes.user_tag_1 = user_tag
        self._capture_operation_results([connectivity_index])
        connectivity_index.fk_from_operation = self.operation_id
        dao.store_entity(connectivity_index)

        return connectivity_index.gid

    def __import_time_series_csv_datatype(self, hrf_folder, connectivity_gid, patient, user_tag):
        path = os.path.join(hrf_folder, self.TIME_SERIES_CSV_FILE)
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=DELIMITER_OPTIONS['comma'])
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
        result = ABCUploader.read_matlab_data(path, self.FC_DATASET_NAME)
        result = result.reshape((result.shape[0], result.shape[1], 1, 1))

        project = dao.get_project_by_id(self.current_project_id)
        user = dao.get_user_by_id(project.fk_admin)

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

    def launch(self, view_model):
        # type: (TumorDatasetImporterModel) -> []
        """
        Download the Tumor Dataset and then import its data
        (currently only the connectivties and pearson coefficients are imported).
        """

        storage_interface = StorageInterface()
        folder = storage_interface.get_tumor_dataset_folder()
        download_path = os.path.join(folder, self.TUMOR_ZIP_FILE_NAME)
        import_path = os.path.join(os.path.dirname(download_path), 'TVB_brain_tumor', 'derivatives', 'TVB')

        if not os.path.exists(import_path):
            was_downloaded = self.__download_tumor_dataset(0, download_path, self.MAXIMUM_DOWNLOAD_RETRIES,
                                                           view_model.tumor_dataset_url)

            if was_downloaded is False:
                return

            storage_interface.unpack_zip(download_path, os.path.dirname(download_path))
            os.remove(download_path)

        for patient in os.listdir(import_path):
            patient_path = os.path.join(import_path, patient)
            if os.path.isdir(patient_path):
                user_tags = os.listdir(patient_path)
                for user_tag in user_tags:
                    datatype_folder = os.path.join(patient_path, user_tag)

                    conn_gid = self.__import_tumor_connectivity(datatype_folder, patient, user_tag, self.storage_path)

                    # The Time Series are invisible in the UI and are imported
                    # just so we can link them with the Pearson Coefficients
                    ts_gid = self.__import_time_series_csv_datatype(datatype_folder, conn_gid, patient, user_tag)
                    self.__import_pearson_coefficients_datatype(datatype_folder, patient, user_tag, ts_gid)

        self.logger.debug("Importing Tumor Dataset has been successfully completed!")

    def __download_tumor_dataset(self, start_byte, file_name, retry_no, tumor_dataset_url):
        if retry_no == 0:
            self.logger.error(
                "The download of the tumor dataset has failed. The maximum number of retries ({}) has been reached!"
                    .format(self.MAXIMUM_DOWNLOAD_RETRIES))
            return False

        resume_header = {'Range': f'bytes={start_byte}'}

        try:
            r = requests.get(tumor_dataset_url, stream=True, headers=resume_header)

            with open(file_name, 'ab') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)

            self.logger.debug("The download of the tumor dataset has been successfully completed!")
            return True

        except Exception:
            self.logger.warning("An unexpected error has appeared while downloading the tumor dataset."
                                " Trying the download again, number of retries left is {}!".format(retry_no - 1))
            time.sleep(self.SLEEP_TIME)
            next_start_byte = Path(file_name).stat().st_size
            return self.__download_tumor_dataset(next_start_byte, file_name, retry_no - 1)
