# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
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

import unittest
from StringIO import StringIO
from tvb.adapters.uploaders.obj.parser import ObjWriter, ObjParser



class ObjFilesTest(unittest.TestCase):

    def test_write_simple(self):
        f = StringIO()
        w = ObjWriter(f)
        w.write([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 2], [0, 1, 3]])
        self.assertTrue(len(f.getvalue()) > 15)


    def test_write_with_normals(self):
        f = StringIO()
        w = ObjWriter(f)
        w.write([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 2], [0, 1, 3]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
                comment="exported from test")
        self.assertTrue(len(f.getvalue()) > 15)


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
        self.assertEqual(vertices, p.vertices)
        self.assertEqual(normals, p.normals)
        # self.assertEqual(triangles, p.faces)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ObjFilesTest))
    return test_suite



if __name__ == "__main__":
    #To run tests individually.
    unittest.main()

