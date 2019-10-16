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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.uploaders.gifti.parser import GIFTIParser, OPTION_READ_METADATA
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException, ParseException
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.datatypes.surfaces import ALL_SURFACES_SELECTION
from tvb.core.neotraits._forms import UploadField, SimpleBoolField, SimpleSelectField
from tvb.interfaces.neocom._h5loader import DirLoader


class GIFTISurfaceImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(GIFTISurfaceImporterForm, self).__init__(prefix, project_id)
        surface_options = {'Specified in the file metadata': OPTION_READ_METADATA}
        surface_options.update(ALL_SURFACES_SELECTION)

        self.file_type = SimpleSelectField(surface_options, self, name='file_type', required=True,
                                           label='Specify file type : ')
        self.data_file = UploadField('.gii', self, name='data_file', required=True,
                                     label='Please select a .gii (LH if cortex)')
        self.data_file_part2 = UploadField('.gii', self, name='data_file_part2',
                                           label="Optionally select 2'nd .gii (RH if cortex)")
        self.should_center = SimpleBoolField(self, name='should_center', default=False,
                                             label='Center surface using vertex means along axes')


class GIFTISurfaceImporter(ABCUploader):
    """
    This importer is responsible for import of surface from GIFTI format (XML file)
    and store them in TVB as Surface.
    """
    _ui_name = "Surface GIFTI"
    _ui_subsection = "gifti_surface_importer"
    _ui_description = "Import a surface from GIFTI"

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return GIFTISurfaceImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [SurfaceIndex]


    def launch(self, file_type, data_file, data_file_part2, should_center=False):
        """
        Execute import operations:
        """
        parser = GIFTIParser(self.storage_path, self.operation_id)
        try:
            surface = parser.parse(data_file, data_file_part2, file_type, should_center=should_center)
            surface.compute_triangle_normals()
            surface.compute_vertex_normals()
            validation_result = surface.validate()

            if validation_result.warnings:
                self.add_operation_additional_info(validation_result.summary())

            surface_idx = SurfaceIndex()
            surface_idx.fill_from_has_traits(surface)

            loader = DirLoader(self.storage_path)
            surface_h5_path = loader.path_for(SurfaceH5, surface_idx.gid)
            with SurfaceH5(surface_h5_path) as surface_h5:
                surface_h5.store(surface)

            return [surface_idx]
        except ParseException as excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)