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

from io import StringIO
from tvb.adapters.uploaders.obj.parser import ObjWriter, ObjParser


class TestObjFiles():

    def test_write_simple(self):
        f = StringIO()
        w = ObjWriter(f)
        w.write([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 2], [0, 1, 3]])
        assert len(f.getvalue()) > 15

    def test_write_with_normals(self):
        f = StringIO()
        w = ObjWriter(f)
        w.write([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 2], [0, 1, 3]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                comment="exported from test")
        assert len(f.getvalue()) > 15

    def test_write_parse_cycle(self):
        f = StringIO()
        w = ObjWriter(f)
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
        triangles = [(0, 1, 2), (0, 1, 3)]
        w.write(vertices, triangles, normals)

        f.seek(0)

        p = ObjParser()
        p.read(f)
        assert vertices == p.vertices
        assert normals == p.normals
        # assert triangles == p.faces

