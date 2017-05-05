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

from tvb.datatypes import mode_decompositions, time_series
from tvb.tests.library.base_testcase import BaseTestCase
        
class ModeDecompositionsTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.mode_decompositions` module.
    """
    
    def test_principalcomponents(self):
        data = numpy.random.random((10, 10, 10, 10))
        ts = time_series.TimeSeries(data=data)
        dt = mode_decompositions.PrincipalComponents(source = ts,
                                                    fractions = numpy.random.random((10, 10, 10)),
                                                    weights = data)
        dt.configure()
        dt.compute_norm_source()
        dt.compute_component_time_series()
        dt.compute_normalised_component_time_series()
        summary = dt.summary_info
        self.assertEqual(summary['Mode decomposition type'], 'PrincipalComponents')
        self.assertTrue(dt.source is not None)
        self.assertEqual(dt.weights.shape, (10, 10, 10, 10))
        self.assertEqual(dt.fractions.shape, (10, 10, 10))
        self.assertEqual(dt.norm_source.shape, (10, 10, 10, 10))
        self.assertEqual(dt.component_time_series.shape, (10, 10, 10, 10))
        self.assertEqual(dt.normalised_component_time_series.shape, (10, 10, 10, 10))
        
        
    def test_independentcomponents(self):
        data = numpy.random.random((10, 10, 10, 10))
        ts = time_series.TimeSeries(data=data)
        n_comp = 5
        dt = mode_decompositions.IndependentComponents(  source = ts,
                                         component_time_series = numpy.random.random((10, n_comp, 10, 10)), 
                                         prewhitening_matrix = numpy.random.random((n_comp, 10, 10, 10)),
                                         unmixing_matrix = numpy.random.random((n_comp, n_comp, 10, 10)),
                                         n_components = n_comp)
        dt.compute_norm_source()
        dt.compute_component_time_series()
        dt.compute_normalised_component_time_series()
        summary = dt.summary_info
        self.assertEqual(summary['Mode decomposition type'], 'IndependentComponents')
        self.assertTrue(dt.source is not None)
        self.assertEqual(dt.mixing_matrix.shape, (0,))
        self.assertEqual(dt.unmixing_matrix.shape, (n_comp, n_comp, 10, 10))
        self.assertEqual(dt.prewhitening_matrix.shape, (n_comp, 10, 10, 10))
        self.assertEqual(dt.norm_source.shape, (10, 10, 10, 10))
        self.assertEqual(dt.component_time_series.shape, (10, 10, n_comp, 10))
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ModeDecompositionsTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 