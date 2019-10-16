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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import uuid
import numpy
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.uploaders.zip_surface.parser import ZipSurfaceParser
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.datatypes.surfaces import make_surface, center_vertices, ALL_SURFACES_SELECTION
from tvb.core.neotraits._forms import UploadField, SimpleSelectField, SimpleBoolField
from tvb.interfaces.neocom._h5loader import DirLoader


class ZIPSurfaceImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(ZIPSurfaceImporterForm, self).__init__(prefix, project_id)
        self.uploaded = UploadField('application/zip', self, name='uploaded', required=True, label='Surface file (zip)')
        self.surface_type = SimpleSelectField(ALL_SURFACES_SELECTION, self, name='surface_type', required=True,
                                              label='Surface type')
        self.zero_based_triangles = SimpleBoolField(self, name='zero_based_triangles', default=True,
                                                    label='Zero based triangles')
        self.should_center = SimpleBoolField(self, name='should_center',
                                             label='Center surface using vertex means along axes')


class ZIPSurfaceImporter(ABCUploader):
    """
    Handler for uploading a Surface Data archive, with files holding
    vertices, normals and triangles to represent a surface data.
    """

    _ui_name = "Surface ZIP"
    _ui_subsection = "zip_surface_importer"
    _ui_description = "Import a Surface from ZIP"
    logger = get_logger(__name__)

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return ZIPSurfaceImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [SurfaceIndex]


    @staticmethod
    def _make_surface(surface_type):

        result = make_surface(surface_type)

        if result is not None:
            return result

        exception_str = "Could not determine surface type (selected option %s)" % surface_type
        raise LaunchException(exception_str)


    def launch(self, uploaded, surface_type, zero_based_triangles=False, should_center=False):
        """
        Execute import operations: unpack ZIP and build Surface object as result.

        :param uploaded: an archive containing the Surface data to be imported
        :param surface_type: a string from the following\: \
                            "Skin Air", "Skull Skin", "Brain Skull", "Cortical Surface", "EEG Cap", "Face"

        :returns: a subclass of `Surface` DataType
        :raises LaunchException: when
                * `uploaded` is missing
                * `surface_type` is invalid
        :raises RuntimeError: when triangles contain an invalid vertex index
        """
        if uploaded is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        self.logger.debug("Start to import surface: '%s' from file: %s" % (surface_type, uploaded))
        try:
            zip_surface = ZipSurfaceParser(uploaded)
        except IOError:
            exception_str = "Did not find the specified ZIP at %s" % uploaded
            raise LaunchException(exception_str)

        # Detect and instantiate correct surface type
        self.logger.debug("Create surface instance")
        surface = self._make_surface(surface_type)
        surface.zero_based_triangles = zero_based_triangles
        if should_center:
            vertices = center_vertices(zip_surface.vertices)
        else:
            vertices = zip_surface.vertices
        surface.vertices = vertices
        if len(zip_surface.normals) != 0:
            surface.vertex_normals = zip_surface.normals
        if zero_based_triangles:
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
            if triangles_min_vertex == -1 and not zero_based_triangles:
                raise LaunchException("Triangles contain a negative vertex index. Maybe you have a ZERO based surface.")
            else:
                raise LaunchException("Your triangles contain a negative vertex index: %d" % triangles_min_vertex)

        no_of_vertices = len(surface.vertices)
        triangles_max_vertex = numpy.amax(surface.triangles)
        if triangles_max_vertex >= no_of_vertices:
            if triangles_max_vertex == no_of_vertices and zero_based_triangles:
                raise LaunchException("Your triangles contain an invalid vertex index: %d. "
                                      "Maybe your surface is NOT ZERO Based." % triangles_max_vertex)
            else:
                raise LaunchException("Your triangles contain an invalid vertex index: %d." % triangles_max_vertex)

        validation_result = surface.validate()

        if validation_result.warnings:
            self.add_operation_additional_info(validation_result.summary())

        self.logger.debug("Surface ready to be stored")

        surf_idx = SurfaceIndex()
        surf_idx.fill_from_has_traits(surface)
        surface.configure()

        loader = DirLoader(self.storage_path)
        surf_path = loader.path_for(SurfaceH5, surf_idx.gid)

        with SurfaceH5(surf_path) as surf_h5:
            surf_h5.store(surface)
            surf_h5.gid.store(uuid.UUID(surf_idx.gid))

        return surf_idx
