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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import unittest
from tvb.simulator.simulator import Simulator
from tvb.simulator.models import Model
from tvb.basic.traits.core import TYPE_REGISTER
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.connectivity import Connectivity


class TraitedInterfaceTests(unittest.TestCase):
    """
    Test class for traits.traited_interface.
    """
    
    
    def test_subclasses_retrieved(self):
        """
        Test that the subclasses for a traited abstract class are correctly retrieved
        """
        subclasses = TYPE_REGISTER.subclasses(Model)
        self.assertTrue(len(subclasses) >= 5)
    
    
    
    def test_sim_interface(self):
        """
        Test that the interface method returns all attributes required for the UI.
        """
        sim = Simulator()
        sim.trait.bound = 'attributes-only'
        current_dict = sim.interface['attributes']
        self.assertFalse(current_dict is None)
        attr = self._get_attr_description(current_dict, 'monitors') 
        self.assertEquals('selectMultiple', attr['type'])
        self.assertEqual('TemporalAverage', attr['default'])
        
        attr = self._get_attr_description(current_dict, 'model')
        self.assertEquals('select', attr['type'])
        self.assertFalse('datatype' in attr)
        self.assertTrue('options' in attr)
        self.assertTrue(len(attr['options']) >= 5)
        for model_attr in attr['options']:
            self.assertTrue(model_attr['name'] is not None)
            self.assertTrue(model_attr['value'] is not None)
            self.assertTrue(model_attr['value'] in model_attr['class'])
            self.assertTrue(len(model_attr['attributes']) >= 3)
            
        attr = self._get_attr_description(current_dict, 'connectivity')
        self.assertTrue(attr['datatype'])
        self.assertEqual(Connectivity.__module__ + "." + Connectivity.__name__, attr['type'])
        attr = self._get_attr_description(current_dict, 'surface')
        self.assertTrue(attr['datatype'])
        self.assertEqual(CorticalSurface.__module__ + "." + CorticalSurface.__name__, attr['type'])
        
        attr = self._get_attr_description(current_dict, 'conduction_speed')
        self.assertEquals(attr['type'], 'float')
        attr = self._get_attr_description(current_dict, 'simulation_length')
        self.assertEquals(attr['type'], 'float')
        self.assertEquals(attr['default'], 1000.0)
        self._validate_list(current_dict)


    def _get_attr_description(self, attributes_list, attribute_name):
        """Find an attribute in the interface, and return its description."""
        for attr_description in attributes_list:
            if attr_description['name'] == attribute_name:
                return attr_description
        self.fail("Could not find attribute "+ str(attribute_name))
        
        
    def _validate_list(self, current_list):
        """ Validate all attributes"""
        for attr_description in current_list:
            if 'name' not in attr_description:
                self.fail("Found attribute without name "+str(attr_description))
            if ('attributes' in attr_description 
                and attr_description['attributes'] 
                and len(attr_description['attributes']) > 0):
                self._validate_list(attr_description['attributes'])
                
                
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TraitedInterfaceTests))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)               
                
