# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Created on Mar 20, 2013

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()
    
    
import numpy
import unittest

from tvb.datatypes import arrays
from tvb.tests.library.base_testcase import BaseTestCase
        
class ArraysTest(BaseTestCase):
    
    
    def test_float_array(self):
        """
        Create a float array, check that shape is correct.
        """
        data = numpy.random.random((10, 10))
        array_dt = arrays.FloatArray()
        array_dt.data = data
        self.assertEqual(array_dt.shape, (10, 10))
        
        
    def test_integer_array(self):
        """
        Create an integer array, check that shape is correct.
        """
        data = numpy.arange(100, dtype=int)
        array_dt = arrays.IntegerArray()
        array_dt.data = data
        self.assertEqual(array_dt.shape, (100,))
        
        
    def test_complex_array(self):
        """
        Create a complex array, check that shape is correct.
        """
        data = numpy.array([numpy.complex(100, 2) for _ in range(100)])
        array_dt = arrays.ComplexArray()
        array_dt.data = data
        self.assertEqual(array_dt.shape, (100,))
        
        
    def test_bool_array(self):
        """
        Create a boolean array, check that shape is correct.
        """
        data = numpy.array([[False for _ in range(12)] for _ in range(10)])
        array_dt = arrays.ComplexArray()
        array_dt.data = data
        self.assertEqual(array_dt.shape, (10, 12))
        
    
    def test_string_array(self):
        """
        Create a string array, check that shape is correct.
        """
        data = numpy.array([['test' for _ in range(12)] for _ in range(10)])
        array_dt = arrays.StringArray()
        array_dt.data = data
        self.assertEqual(array_dt.shape, (10, 12))
        
        
    def test_position_array(self):
        """
        Create a position array, check that shape is correct.
        """
        data = numpy.random.random((10, 10))
        array_dt = arrays.PositionArray(coordinate_system="test_system",
                                        coordinate_space="test_space")
        array_dt.data = data
        self.assertEqual(array_dt.shape, (10, 10))
        self.assertEqual(array_dt.coordinate_space, "test_space")
        self.assertEqual(array_dt.coordinate_system, "test_system")
        
        
    def test_orientation_array(self):
        """
        Create an orientation array, check that shape is correct.
        """
        data = numpy.random.random((10, 10))
        array_dt = arrays.OrientationArray(coordinate_system_or="test_system")
        array_dt.data = data
        self.assertEqual(array_dt.shape, (10, 10))
        self.assertEqual(array_dt.coordinate_system_or, "test_system")
        
        
    def test_index_array(self):
        """
        Create an index array, check that shape is correct.
        """
        target_data = numpy.random.random((10, 3))
        target_array = arrays.FloatArray()
        target_array.data = target_data
        array_dt = arrays.IndexArray(target=target_array)
        array_dt.data = numpy.arange(30).reshape((10, 3))
        self.assertEqual(array_dt.shape, (10, 3))
        self.assertEqual(array_dt.target.shape, (10, 3))
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ArraysTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 