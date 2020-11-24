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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

from tvb.adapters.uploaders.gifti.parser import GIFTIParser, OPTION_READ_METADATA
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Attr
from tvb.core.adapters.exceptions import LaunchException, ParseException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.datatypes.db.surface import SurfaceIndex, ALL_SURFACES_SELECTION
from tvb.core.neotraits.forms import SelectField, TraitUploadField, BoolField
from tvb.core.neocom import h5
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str


class GIFTISurfaceImporterModel(UploaderViewModel):
    KEY_OPTION_READ_METADATA = 'Specified in the file metadata'
    surface_types = ALL_SURFACES_SELECTION.copy()
    surface_types[KEY_OPTION_READ_METADATA] = OPTION_READ_METADATA

    file_type = Str(
        label='Specify file type : ',
        choices=tuple(surface_types.values()),
        default=tuple(surface_types.values())[0]
    )

    data_file = Str(
        label='Please select a .gii (LH if cortex)'
    )

    data_file_part2 = Str(
        required=False,
        label="Optionally select 2'nd .gii (RH if cortex)"
    )

    should_center = Attr(
        field_type=bool,
        required=False,
        default=False,
        label='Center surface using vertex means along axes'
    )


class GIFTISurfaceImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(GIFTISurfaceImporterForm, self).__init__(project_id)

        self.file_type = SelectField(GIFTISurfaceImporterModel.file_type, self.project_id, name='file_type',
                                     choices=GIFTISurfaceImporterModel.surface_types)
        self.data_file = TraitUploadField(GIFTISurfaceImporterModel.data_file, '.gii', self.project_id,
                                          'data_file', self.temporary_files)
        self.data_file_part2 = TraitUploadField(GIFTISurfaceImporterModel.data_file_part2, '.gii', self.project_id,
                                                'data_file_part2', self.temporary_files)
        self.should_center = BoolField(GIFTISurfaceImporterModel.should_center, self.project_id, name='should_center')

    @staticmethod
    def get_view_model():
        return GIFTISurfaceImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.gii',
            'data_file_part2': '.gii'
        }


class GIFTISurfaceImporter(ABCUploader):
    """
    This importer is responsible for import of surface from GIFTI format (XML file)
    and store them in TVB as Surface.
    """
    _ui_name = "Surface GIFTI"
    _ui_subsection = "gifti_surface_importer"
    _ui_description = "Import a surface from GIFTI"

    def get_form_class(self):
        return GIFTISurfaceImporterForm

    def get_output(self):
        return [SurfaceIndex]

    def launch(self, view_model):
        # type: (GIFTISurfaceImporterModel) -> [SurfaceIndex]
        """
        Execute import operations:
        """
        parser = GIFTIParser(self.storage_path, self.operation_id)
        try:
            surface = parser.parse(view_model.data_file, view_model.data_file_part2, view_model.file_type,
                                   should_center=view_model.should_center)
            surface.compute_triangle_normals()
            surface.compute_vertex_normals()
            validation_result = surface.validate()

            if validation_result.warnings:
                self.add_operation_additional_info(validation_result.summary())
            surface_idx = h5.store_complete(surface, self.storage_path)
            return [surface_idx]
        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)
