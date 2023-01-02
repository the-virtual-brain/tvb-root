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

import numpy as np
from tvb.adapters.uploaders.obj.parser import ObjParser
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException


class ObjSurface(object):
    """
    Represents a surface compatible with tvb.
    self.triangles , self.vertices , self.normals are numpy arrays shaped (n, 3)
    self.triangles contains indices.
    """

    def __init__(self, obj_file):
        """
        Create a surface from an obj file
        """
        self.logger = get_logger(__name__)

        try:
            obj = ObjParser()
            obj.read(obj_file)

            self.triangles = []
            self.vertices = obj.vertices
            self.normals = [(0.0, 0.0, 0.0)] * len(self.vertices)
            self.have_normals = len(obj.normals)

            for face in obj.faces:
                triangles = self._triangulate(face)
                for v_idx, t_idx, n_idx in triangles:
                    self.triangles.append(v_idx)
                    if n_idx != -1:
                        # last normal index wins
                        # alternative: self.normals[v_idx] += obj.normals[n_idx]
                        # The correct behaviour is to duplicate the vertex
                        # self.vertices.append(self.vertices[v_idx])
                        # self.tex_coords.append(self.tex_coords[v_idx])
                        self.normals[v_idx] = obj.normals[n_idx]
            # checks
            if not self.vertices or not self.triangles:
                raise ParseException("No geometry data in file.")
            self._to_numpy()
        except ValueError as ex:
            self.logger.exception(" Error in obj")
            raise ParseException(str(ex))


    def _triangulate(self, face):
        """
        Triangulate a quad. Higher order will get truncated.
        """
        if len(face) > 4:
            self.logger.warning("truncated face to a quad")
        triangles = face[:3]
        if len(face) == 4:
            triangles += [face[0], face[2], face[3]]
        return triangles


    def _to_numpy(self):
        self.vertices = np.array(self.vertices)

        if self.have_normals:
            self.normals = np.array(self.normals)
            # normalise to unit vectors
            for k in range(len(self.normals)):
                self.normals[k, :] = self.normals[k, :] / np.sqrt(np.sum(self.normals[k, :] ** 2, axis=0))
        else:
            self.normals = None

        self.triangles = np.array(self.triangles).reshape((len(self.triangles) // 3, 3))
