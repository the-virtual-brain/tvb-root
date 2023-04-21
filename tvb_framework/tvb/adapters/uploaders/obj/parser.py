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
Handlers for reading/writing from/into OBJ format Surfaces are defined in this module.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from tvb.basic.logger.builder import get_logger



class ObjParser(object):
    """
    This class reads geometry from a simple wavefront obj file.
    ``self.vertices``, ``self.tex_coords``, ``self.normals`` are lists of vectors represented as tuples
    ``self.faces`` is a list of faces. A face is a list of vertex info.
    Vertex info is a tuple vertex_index, tex_index, normal_index.
    """


    def __init__(self):
        self.logger = get_logger(__name__)
        self.vertices = []
        self.normals = []
        self.tex_coords = []
        self.faces = []


    def parse_v(self, args):
        self.vertices.append(tuple(float(x) for x in args[:3]))


    def parse_vn(self, args):
        self.normals.append(tuple(float(x) for x in args[:3]))


    def parse_vt(self, args):
        self.tex_coords.append(tuple(float(x) for x in args[:3]))


    def parse_f(self, args):
        if len(args) < 3:
            raise ValueError("Require at least 3 vertices per face")

        face = []

        for v in args:
            indices = []

            for j in v.split('/'):
                if j == '':
                    indices.append(-1)
                else:
                    indices.append(int(j) - 1)

            while len(indices) < 3:
                indices.append(-1)

            face.append(indices)
        self.faces.append(face)


    def read(self, obj_file):
        line_nr = -1
        try:
            for line_nr, line in enumerate(obj_file):
                line = line.strip()
                if line == "" or line[0] == '#':
                    continue
                tokens = line.split()
                data_type = tokens[0]
                datatype_parser = getattr(self, "parse_%s" % data_type, None)

                if datatype_parser is not None:
                    datatype_parser(tokens[1:])
                else:
                    self.logger.warning("Unsupported token type %s" % data_type)
        except ValueError as ex:
            raise ValueError("%s at line %d" % (ex, line_nr))



class ObjWriter(object):

    def __init__(self, obj_file):
        self.logger = get_logger(__name__)
        self.file = obj_file


    def _write_vector(self, type_code, v):
        v_str = ' '.join(str(e) for e in v)
        self.file.write('%s %s\n' % (type_code, v_str))


    def _write_face(self, v_info, write_normals):
        fs = []
        for info in v_info:
            info += 1
            if not write_normals:
                s = '%d' % info
            else:
                s = '%d//%d' % (info, info)
            fs.append(s)
        self.file.write('f %s \n' % ' '.join(fs))


    def write(self, vertices, faces, normals=None, comment=''):
        """
        :param vertices:, :param normals: are lists of vectors or ndarrays of shape (n,3)
        :param faces: A face is a list of 3 vertex indices.
        Normal indices not supported. Texture uv's not supported.
        This method does not yet validate the input, so send valid data.
        """
        self.file.write('# %s\n' % comment)
        for v in vertices:
            self._write_vector('v', v)

        self.file.write('\n')

        if normals is not None:
            for v in normals:
                self._write_vector('vn', v)
            self.file.write('\n')

        for f in faces:
            self._write_face(f, normals is not None)
