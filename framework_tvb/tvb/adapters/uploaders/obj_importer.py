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
"""

import uuid
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.adapters.uploaders.obj.surface import ObjSurface
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.storage import transactional
from tvb.datatypes.surfaces import make_surface, center_vertices, ALL_SURFACES_SELECTION
from tvb.core.neotraits._forms import SimpleSelectField, UploadField, SimpleBoolField
from tvb.interfaces.neocom._h5loader import DirLoader


class ObjSurfaceImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(ObjSurfaceImporterForm, self).__init__(prefix, project_id)

        self.surface_type = SimpleSelectField(ALL_SURFACES_SELECTION, self, name='surface_type', required=True,
                                              label='Specify file type :')
        self.data_file = UploadField('.obj', self, name='data_file', required=True, label='Please select file to import')
        self.should_center = SimpleBoolField(self, name='should_center', default=False,
                                             label='Center surface using vertex means along axes')


class ObjSurfaceImporter(ABCUploader):
    """
    This imports geometry data stored in wavefront obj format
    """
    _ui_name = "Surface OBJ"
    _ui_subsection = "obj_importer"
    _ui_description = "Import geometry data stored in wavefront obj format"

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return ObjSurfaceImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [SurfaceIndex]


    @transactional
    def launch(self, surface_type, data_file, should_center=False):
        """
        Execute import operations:
        """
        try:
            surface = make_surface(surface_type)
            if surface is None:
                raise ParseException("Could not determine surface type! %s" % surface_type)

            surface.zero_based_triangles = True

            with open(data_file) as f:
                obj = ObjSurface(f)

            if should_center:
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

            surface_idx = SurfaceIndex()
            surface_idx.fill_from_has_traits(surface)

            loader = DirLoader(self.storage_path)
            surface_h5_path = loader.path_for(SurfaceH5, surface_idx.gid)

            with SurfaceH5(surface_h5_path) as surface_h5:
                surface_h5.store(surface)
                surface_h5.gid.store(uuid.UUID(surface_idx.gid))

            return [surface_idx]

        except ParseException as excep:
            self.log.exception(excep)
            raise LaunchException(excep)