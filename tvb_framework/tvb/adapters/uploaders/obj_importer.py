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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from tvb.adapters.uploaders.obj.surface import ObjSurface
from tvb.basic.neotraits.api import Attr, EnumAttr
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.storage import transactional
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.surfaces import make_surface, center_vertices, SurfaceTypesEnum
from tvb.core.neotraits.forms import BoolField, TraitUploadField, SelectField


class ObjSurfaceImporterModel(UploaderViewModel):
    surface_type = EnumAttr(
        label='Specify file type :',
        default=SurfaceTypesEnum.CORTICAL_SURFACE
    )

    data_file = Str(
        label='Please select file to import'
    )

    should_center = Attr(
        field_type=bool,
        required=False,
        default=False,
        label='Center surface using vertex means along axes'
    )


class ObjSurfaceImporterForm(ABCUploaderForm):

    def __init__(self):
        super(ObjSurfaceImporterForm, self).__init__()

        self.surface_type = SelectField(ObjSurfaceImporterModel.surface_type, name='surface_type')
        self.data_file = TraitUploadField(ObjSurfaceImporterModel.data_file, '.obj', 'data_file')
        self.should_center = BoolField(ObjSurfaceImporterModel.should_center, name='should_center')

        del self.surface_type.choices[-1]

    @staticmethod
    def get_view_model():
        return ObjSurfaceImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.obj'
        }


class ObjSurfaceImporter(ABCUploader):
    """
    This imports geometry data stored in wavefront obj format
    """
    _ui_name = "Surface OBJ"
    _ui_subsection = "obj_importer"
    _ui_description = "Import geometry data stored in wavefront obj format"

    def get_form_class(self):
        return ObjSurfaceImporterForm

    def get_output(self):
        return [SurfaceIndex]

    @transactional
    def launch(self, view_model):
        # type: (ObjSurfaceImporterModel) -> [SurfaceIndex]
        """
        Execute import operations:
        """
        try:
            surface = make_surface(view_model.surface_type.value)
            if surface is None:
                raise ParseException("Could not determine surface type! %s" % view_model.surface_type)

            surface.zero_based_triangles = True

            with open(view_model.data_file) as f:
                obj = ObjSurface(f)

            if view_model.should_center:
                vertices = center_vertices(obj.vertices)
            else:
                vertices = obj.vertices

            surface.vertices = vertices
            surface.triangles = obj.triangles

            if obj.have_normals:
                self.log.debug("OBJ came with normals included")
                surface.vertex_normals = obj.normals
            else:
                self.log.warning("OBJ came without normals. We will try to compute them...")

            surface.number_of_vertices = surface.vertices.shape[0]
            surface.number_of_triangles = surface.triangles.shape[0]
            surface.compute_triangle_normals()
            surface.compute_vertex_normals()

            validation_result = surface.validate()
            if validation_result.warnings:
                self.add_operation_additional_info(validation_result.summary())

            return self.store_complete(surface)

        except ParseException as excep:
            self.log.exception(excep)
            raise LaunchException(excep)
