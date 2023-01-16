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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy

from tvb.adapters.uploaders.zip_surface.parser import ZipSurfaceParser
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Attr, EnumAttr
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.neotraits.forms import TraitUploadField, SelectField, BoolField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str
from tvb.datatypes.surfaces import make_surface, center_vertices, SurfaceTypesEnum


class ZIPSurfaceImporterModel(UploaderViewModel):
    uploaded = Str(
        label='Surface file (zip)'
    )

    surface_type = EnumAttr(
        default=SurfaceTypesEnum.CORTICAL_SURFACE,
        label='Surface type'
    )

    zero_based_triangles = Attr(
        field_type=bool,
        required=False,
        default=True,
        label='Zero based triangles'
    )

    should_center = Attr(
        field_type=bool,
        required=False,
        label='Center surface using vertex means along axes'
    )


class ZIPSurfaceImporterForm(ABCUploaderForm):

    def __init__(self):
        super(ZIPSurfaceImporterForm, self).__init__()
        self.uploaded = TraitUploadField(ZIPSurfaceImporterModel.uploaded, '.zip', 'uploaded')
        self.surface_type = SelectField(ZIPSurfaceImporterModel.surface_type, 'surface_type')
        self.zero_based_triangles = BoolField(ZIPSurfaceImporterModel.zero_based_triangles, name='zero_based_triangles')
        self.should_center = BoolField(ZIPSurfaceImporterModel.should_center, name='should_center')

        del self.surface_type.choices[-1]

    @staticmethod
    def get_view_model():
        return ZIPSurfaceImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'uploaded': '.zip'
        }


class ZIPSurfaceImporter(ABCUploader):
    """
    Handler for uploading a Surface Data archive, with files holding
    vertices, normals and triangles to represent a surface data.
    """

    _ui_name = "Surface ZIP"
    _ui_subsection = "zip_surface_importer"
    _ui_description = "Import a Surface from ZIP"
    logger = get_logger(__name__)

    def get_form_class(self):
        return ZIPSurfaceImporterForm

    def get_output(self):
        return [SurfaceIndex]

    @staticmethod
    def _make_surface(surface_type):

        result = make_surface(surface_type)

        if result is not None:
            return result

        exception_str = "Could not determine surface type (selected option %s)" % surface_type
        raise LaunchException(exception_str)

    def launch(self, view_model):
        # type: (ZIPSurfaceImporterModel) -> [SurfaceIndex]
        """
        Execute import operations: unpack ZIP and build Surface object as result
        :raises LaunchException: when
                * `uploaded` is missing
                * `surface_type` is invalid
        :raises RuntimeError: when triangles contain an invalid vertex index
        """
        if view_model.uploaded is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        self.logger.debug(
            "Start to import surface: '%s' from file: %s" % (view_model.surface_type, view_model.uploaded))
        try:
            zip_surface = ZipSurfaceParser(view_model.uploaded)
        except IOError:
            exception_str = "Did not find the specified ZIP at %s" % view_model.uploaded
            raise LaunchException(exception_str)

        # Detect and instantiate correct surface type
        self.logger.debug("Create surface instance")
        surface = self._make_surface(view_model.surface_type.value)
        surface.zero_based_triangles = view_model.zero_based_triangles
        if view_model.should_center:
            vertices = center_vertices(zip_surface.vertices)
        else:
            vertices = zip_surface.vertices
        surface.vertices = vertices
        if len(zip_surface.normals) != 0:
            surface.vertex_normals = zip_surface.normals
        if view_model.zero_based_triangles:
            surface.triangles = zip_surface.triangles
        else:
            surface.triangles = zip_surface.triangles - 1

        if zip_surface.bi_hemispheric:
            self.logger.info("Hemispheres detected")

        surface.hemisphere_mask = zip_surface.hemisphere_mask
        surface.compute_triangle_normals()

        # Now check if the triangles of the surface are valid
        triangles_min_vertex = numpy.amin(surface.triangles)
        if triangles_min_vertex < 0:
            if triangles_min_vertex == -1 and not view_model.zero_based_triangles:
                raise LaunchException("Triangles contain a negative vertex index. Maybe you have a ZERO based surface.")
            else:
                raise LaunchException("Your triangles contain a negative vertex index: %d" % triangles_min_vertex)

        no_of_vertices = len(surface.vertices)
        triangles_max_vertex = numpy.amax(surface.triangles)
        if triangles_max_vertex >= no_of_vertices:
            if triangles_max_vertex == no_of_vertices and view_model.zero_based_triangles:
                raise LaunchException("Your triangles contain an invalid vertex index: %d. "
                                      "Maybe your surface is NOT ZERO Based." % triangles_max_vertex)
            else:
                raise LaunchException("Your triangles contain an invalid vertex index: %d." % triangles_max_vertex)

        validation_result = surface.validate()

        if validation_result.warnings:
            self.add_operation_additional_info(validation_result.summary())

        surface.configure()
        self.logger.debug("Surface ready to be stored")

        return self.store_complete(surface)
