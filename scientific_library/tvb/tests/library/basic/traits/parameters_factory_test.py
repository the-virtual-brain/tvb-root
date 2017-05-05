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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()
    
import unittest
from tvb.datatypes import arrays
from tvb.basic.traits.parameters_factory import get_traited_instance_for_name, get_traited_subclasses
from tvb.basic.traits.types_mapped import Array
from tvb.tests.library.base_testcase import BaseTestCase


class ParametersFactoryTest(BaseTestCase):
    
    def test_traitedsubclassed(self):
        """
        Tests successful creation of traited classes.
        """
        # We imported array so we should have all these traited classes registered
        expected = [
                    'IntegerArray',
                    'StringArray', 'PositionArray',
                    'IndexArray',

                    'BoolArray', 'OrientationArray',


                    'FloatArray',
                    'ComplexArray', 'Array', 'SparseMatrix']
        subclasses = get_traited_subclasses(Array)
        for key in expected:
            self.assertTrue(key in subclasses)
            
            
    def test_get_traited_instance(self):
        """
        Try to create an instance of a class using the traited method.
        """
        inst = get_traited_instance_for_name("StringArray", Array, {})
        self.assertTrue(isinstance(inst, arrays.StringArray))



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ParametersFactoryTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 