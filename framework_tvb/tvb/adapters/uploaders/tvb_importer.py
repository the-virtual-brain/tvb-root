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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import shutil
import zipfile
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.core.services.exceptions import ImportException
from tvb.core.services.import_service import ImportService
from tvb.core.entities.storage import dao
from tvb.core.entities.file.hdf5_storage_manager import HDF5StorageManager
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.files_update_manager import FilesUpdateManager


class TVBImporterModel(UploaderViewModel):
    data_file = Str(
        label='Please select file to import (h5 or zip)'
    )


class TVBImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(TVBImporterForm, self).__init__(project_id)

        self.data_file = TraitUploadField(TVBImporterModel.data_file, ('.zip', '.h5'), self.project_id,
                                          'data_file', self.temporary_files)

    @staticmethod
    def get_view_model():
        return TVBImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': ('.zip', '.h5')
        }


class TVBImporter(ABCUploader):
    """
    This importer is responsible for loading of data types exported from other systems
    in TVB format (simple H5 file or ZIP file containing multiple H5 files)
    """
    _ui_name = "TVB HDF5 / ZIP"
    _ui_subsection = "tvb_datatype_importer"
    _ui_description = "Upload H5 file with TVB generic entity"

    def get_form_class(self):
        return TVBImporterForm

    def get_output(self):
        return []

    def _prelaunch(self, operation, view_model, uid=None, available_disk_space=0):
        """
        Overwrite method in order to return the correct number of stored datatypes.
        """
        self.nr_of_datatypes = 0
        msg, _ = ABCUploader._prelaunch(self, operation, view_model, uid, available_disk_space)
        return msg, self.nr_of_datatypes

    def launch(self, view_model):
        # type: (TVBImporterModel) -> []
        """
        Execute import operations: unpack ZIP, build and store generic DataType objects.
        :raises LaunchException: when data_file is None, nonexistent, or invalid \
                    (e.g. incomplete meta-data, not in ZIP / HDF5 format etc. )
        """
        if view_model.data_file is None:
            raise LaunchException("Please select file which contains data to import")

        service = ImportService()
        if os.path.exists(view_model.data_file):
            if zipfile.is_zipfile(view_model.data_file):
                current_op = dao.get_operation_by_id(self.operation_id)

                # Creates a new TMP folder where to extract data
                tmp_folder = os.path.join(self.storage_path, "tmp_import")
                FilesHelper().unpack_zip(view_model.data_file, tmp_folder)
                operations = service.import_project_operations(current_op.project, tmp_folder)
                shutil.rmtree(tmp_folder)
                self.nr_of_datatypes += len(operations)

            else:
                # upgrade file if necessary
                file_update_manager = FilesUpdateManager()
                file_update_manager.upgrade_file(view_model.data_file)

                folder, h5file = os.path.split(view_model.data_file)
                manager = HDF5StorageManager(folder, h5file)
                if manager.is_valid_hdf5_file():
                    datatype = None
                    try:
                        datatype = service.load_datatype_from_file(view_model.data_file, self.operation_id)
                        service.check_import_references(view_model.data_file, datatype)
                        service.store_datatype(datatype, view_model.data_file)
                        self.nr_of_datatypes += 1
                    except ImportException as excep:
                        self.log.exception(excep)
                        if datatype is not None:
                            target_path = h5.path_for_stored_index(datatype)
                            if os.path.exists(target_path):
                                os.remove(target_path)
                        raise LaunchException("Invalid file received as input. " + str(excep))
                else:
                    raise LaunchException("Uploaded file: %s is neither in ZIP or HDF5 format" % view_model.data_file)

        else:
            raise LaunchException("File: %s to import does not exists." % view_model.data_file)
