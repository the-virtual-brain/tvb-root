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

from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.obj.surface import ObjSurface
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional, dao
from tvb.datatypes.surfaces import ALL_SURFACES_SELECTION, FACE, Surface, make_surface, center_vertices


class ObjSurfaceImporter(ABCUploader):
    """
    This imports geometry data stored in wavefront obj format
    """
    _ui_name = "Surface OBJ"
    _ui_subsection = "obj_importer"
    _ui_description = "Import geometry data stored in wavefront obj format"


    def get_upload_input_tree(self):
        """
        Take as input an obj file
        """
        return [{'name': 'surface_type', 'type': 'select',
                 'label': 'Specify file type : ', 'required': True,
                 'options': ALL_SURFACES_SELECTION,
                 'default': FACE},

                {'name': 'data_file', 'type': 'upload', 'required_type': '.obj',
                 'label': 'Please select file to import', 'required': True},

                {'name': 'should_center', 'type': 'bool', 'default': False,
                 'label': 'Center surface using vertex means along axes'}]
        
        
    def get_output(self):
        return [Surface]


    @transactional
    def launch(self, surface_type, data_file, should_center=False):
        """
        Execute import operations:
        """
        try:
            surface = make_surface(surface_type)
            if surface is None:
                raise ParseException("Could not determine surface type! %s" % surface_type)

            surface.storage_path = self.storage_path
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

            validation_result = surface.validate()

            if validation_result.warnings:
                self.add_operation_additional_info(validation_result.summary())

            return [surface]

        except ParseException as excep:
            self.log.exception(excep)
            raise LaunchException(excep)