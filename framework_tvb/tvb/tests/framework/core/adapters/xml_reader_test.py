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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import unittest
from tvb.basic.profile import TvbProfile
from tvb.core.adapters import xml_reader
from tvb.core.adapters.introspector import Introspector
from tvb.core.adapters.exceptions import XmlParserException
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
import tvb.tests.framework.adapters as adapters_init



class XML_Reader_Test(TransactionalTestCase):
    """
    This is a test class for the tvb.core.adapters.xml_reader module.
    """


    def setUp(self):
        """
        This method sets up the necessary paths for testing.
        """
        self.folder_path = os.path.dirname(__file__)
        #tvb.tests.framework path
        core_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.old_path = TvbProfile.current.web.CURRENT_DIR
        TvbProfile.current.web.CURRENT_DIR = os.path.dirname(core_path)
        adapters_init.__xml_folders__ = [os.path.join('core', 'adapters')]
        self.introspector = Introspector("tvb.tests.framework")
        self.introspector.introspect(True)
        xml_group_path = os.path.join('core', 'adapters', "test_group.xml")
        algo_group = dao.find_group('tvb.tests.framework.adapters.testgroupadapter', 'TestGroupAdapter', xml_group_path)
        self.xml_group_adapter = ABCAdapter.build_adapter(algo_group)
        self.xml_group_reader = self.xml_group_adapter.xml_reader


    def tearDown(self):
        """
        Clean-up tests data (xml folders)
        """
        TvbProfile.current.web.CURRENT_DIR = self.old_path
        if hasattr(adapters_init, '__xml_folders__'):
            # Since TransactionalTestCase makes sure the tearDown is called even in setUp or test fails we need
            # to check if this was really added.
            del adapters_init.__xml_folders__


    def test_algorithm_group_attributes(self):
        """
        Tests the attributes of an algorithm group
        """
        self.assertEqual(self.xml_group_adapter.xml_reader.root_name, "simple")
        self.assertEqual(self.xml_group_adapter.xml_reader.get_type(),
                         "tvb.tests.framework.adapters.testgroupadapter.TestGroupAdapter")
        self.assertEqual(self.xml_group_adapter.xml_reader.get_ui_name(), "Simple Python Analyzers")
        self.assertTrue("externals/BCT" in self.xml_group_adapter.xml_reader.get_additional_path())


    def test_code_for_algorithm(self):
        """
        Tests the code that has to be executed for an algorithm
        """
        self.assertEqual(self.xml_group_adapter.get_call_code("CC"),
                         "cross_correlation(data1, data2, chan1, chan2, mode)")


    def test_import_for_algorithm(self):
        """
        Tests the code that has to be imported for an algorithm
        """
        self.assertEqual(self.xml_group_adapter.get_import_code("CC"), "tvb.analyzers.simple_analyzers")


    def test_inputs_for_algorithm(self):
        """
        Tests the inputs for an algorithm
        """
        inputs = self.xml_group_adapter.xml_reader.get_inputs("CC")
        self.check_inputs(inputs)


    def test_outputs_for_algorithm(self):
        """
        Tests the outputs for an algorithm
        """
        outputs = self.xml_group_adapter.xml_reader.get_outputs("CC")
        self.check_outputs(outputs)


    def test_get_interface(self):
        """
        Tests the interface for an algorithm group
        """
        interface = self.xml_group_adapter.get_input_tree()
        self.assertEqual(len(interface), 1, "Didnt parse correctly.")
        tree_root = interface[0]
        self.assertEqual(tree_root[xml_reader.ATT_NAME], "simple")
        self.assertEqual(tree_root[xml_reader.ATT_LABEL], "Analysis Algorithm:")
        self.assertTrue(tree_root[xml_reader.ATT_REQUIRED])
        self.assertEqual(tree_root[xml_reader.ATT_TYPE], xml_reader.TYPE_SELECT)
        options = tree_root[xml_reader.ELEM_OPTIONS]
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0][xml_reader.ATT_VALUE], "CC")
        self.assertEqual(options[0][xml_reader.ATT_NAME], "Cross Correlation")
        attributes = options[0][xml_reader.ATT_ATTRIBUTES]
        self.assertEqual(len(attributes), 3, "Didnt parse correctly.")
        self.check_inputs(attributes)
        outputs = options[0][xml_reader.ELEM_OUTPUTS]
        self.assertEqual(len(outputs), 2, "Didnt parse correctly.")
        self.check_outputs(outputs)


    def test_get_algorithms_dictionary(self):
        """
        Tests the list of algorithms for an algorithm group
        """
        algorithms = self.xml_group_adapter.xml_reader.get_algorithms_dictionary()
        self.assertEqual(len(algorithms), 1, "Didn't parse correctly.")
        algorithm = algorithms["CC"]
        self.assertEqual(algorithm[xml_reader.ATT_IDENTIFIER], "CC")
        self.assertEqual(algorithm[xml_reader.ATT_CODE], "cross_correlation(data1, data2, chan1, chan2, mode)")
        self.assertEqual(algorithm[xml_reader.ATT_IMPORT], "tvb.analyzers.simple_analyzers")
        self.assertEqual(algorithm[xml_reader.ATT_NAME], "Cross Correlation")
        #check inputs
        inputs = algorithms["CC"][xml_reader.INPUTS_KEY]
        new_inputs = []
        for inp in inputs.keys():
            new_inputs.append(inputs[inp])
        self.check_inputs(new_inputs)
        #check outputs
        outputs = algorithms["CC"][xml_reader.OUTPUTS_KEY]
        self.check_outputs(outputs)


    def test_get_all_outputs(self):
        """
        Tests the all outputs of the algorithms from an algorithm group
        """
        outputs = self.xml_group_reader.get_all_outputs()
        self.assertEqual(len(outputs), 2, "Didnt parse correctly.")
        self.check_outputs(outputs)


    def test_no_algorithm_node(self):
        """
        This test trys to load an invalid xml. The xml doesn't contains any
        algorithm node.
        """
        try:
            xml_group_path = os.path.join(self.folder_path, "no_algorithm_node.xml")
            xml_reader.XMLGroupReader.get_instance(xml_group_path)
            self.fail("Test should fail. The xml doesn't contain algorithms")
        except XmlParserException:
            #OK, do nothing
            pass


    def test_no_code_node(self):
        """
        This test trys to load an invalid xml. The xml doesn't contains all
        the required nodes. The missing node is code from one of the algorithms.
        """
        xml_group_path = os.path.join(self.folder_path, "no_code_node.xml")
        self.assertRaises(XmlParserException, xml_reader.XMLGroupReader, xml_group_path)
        #The call should fail. 
        #One of the algorithms doesn't contain required attribute 'code'


    def test_missing_required_attribute(self):
        """
        This test trys to load an invalid xml. The xml doesn't contains all
        the required attributes. The missing attribute is 'name' from one of the algorithms.
        """
        try:
            xml_group_path = os.path.join(self.folder_path, "missing_required_attribute.xml")
            xml_reader.XMLGroupReader.get_instance(xml_group_path)
            self.fail("Test should fail. One of the algorithms doesn't have required 'name' att")
        except XmlParserException:
            #OK, do nothing
            pass


    def test_invalid_schema_url(self):
        """
        Test that when XML schema can not be found (e.g. due to no internet connection, 
        or server down) the XML is still read.
        """
        xml_group_path = os.path.join(self.folder_path, "test_invalid_schema_url.xml")
        self.xml_group_reader = xml_reader.XMLGroupReader.get_instance(xml_group_path)
        self.test_inputs_for_algorithm()
        self.test_outputs_for_algorithm()


    def check_inputs(self, inputs_list):
        """
        The given list of inputs is expected to be for the "CC" algorithm.
        """
        inputs_dict = {}
        for input_ in inputs_list:
            inputs_dict[input_[xml_reader.ATT_NAME]] = input_

        expected_conditions = {'fields': ['datatype_class._nr_dimensions'],
                               'values': ['2'], 'operations': ['==']}
        self._check_input(inputs_dict, "data1", True, "tvb.datatypes.arrays.MappedArray", "First dataset:",
                          "First set of signals", "data", "default_data", None, expected_conditions)

        self._check_input(inputs_dict, "chan1", False, "int", "First channel index:", None, None, None, None, None)

        expected_options = {'valid': 'Valid', 'same': 'Same', 'full': 'Full'}
        self._check_input(inputs_dict, "mode", True, "select", "Mode:", "Flag that indicates the size of the output",
                          None, "full", expected_options, None)


    def _check_input(self, all_inputs, input_name, is_required, expected_type, expected_label, expected_description,
                     expected_field, expected_default, expected_options, expected_conditions):
        """Validate one input"""

        input_ = all_inputs[input_name]

        self.assertEqual(input_[xml_reader.ATT_NAME], input_name)
        self.assertEqual(input_[xml_reader.ATT_LABEL], expected_label)
        self.assertEqual(input_[xml_reader.ATT_TYPE], expected_type)

        if is_required:
            self.assertTrue(input_[xml_reader.ATT_REQUIRED])
        else:
            self.assertFalse(input_[xml_reader.ATT_REQUIRED])

        if expected_description is None:
            self.assertFalse(xml_reader.ATT_DESCRIPTION in input_.keys())
        else:
            self.assertEqual(input_[xml_reader.ATT_DESCRIPTION], expected_description)

        if expected_field is None:
            self.assertFalse(xml_reader.ATT_FIELD in input_.keys())
        else:
            self.assertEqual(input_[xml_reader.ATT_FIELD], expected_field)

        if expected_default is None:
            self.assertFalse(xml_reader.ATT_DEFAULT in input_.keys())
        else:
            self.assertEqual(input_[xml_reader.ATT_DEFAULT], expected_default)

        if expected_options is None:
            self.assertFalse(xml_reader.ELEM_OPTIONS in input_.keys())
        else:
            options = input_[xml_reader.ELEM_OPTIONS]
            for opt in options:
                self.assertTrue(opt[xml_reader.ATT_VALUE] in expected_options)
                self.assertEquals(opt[xml_reader.ATT_NAME], expected_options[opt[xml_reader.ATT_VALUE]])

        if expected_conditions is None:
            self.assertFalse(xml_reader.ELEM_CONDITIONS in input_.keys())
        else:
            ops_filter = input_[xml_reader.ELEM_CONDITIONS]
            for key_attr in expected_conditions.keys():
                actual_values = getattr(ops_filter, key_attr)
                self.assertEqual(len(actual_values), len(expected_conditions[key_attr]))
                for att in actual_values:
                    self.assertTrue(att in expected_conditions[key_attr])


    def check_outputs(self, outputs):
        """
        The given list of outputs is expected to be for the "CC" algorithm.
        """
        self.assertEqual(len(outputs), 2, "Didnt parse correctly.")
        #first output
        self.assertEqual(outputs[0][xml_reader.ATT_TYPE], "tvb.datatypes.arrays.MappedArray")
        fields = outputs[0][xml_reader.ELEM_FIELD]
        self.assertEqual(len(fields), 3, "Expected 3 fields.")
        fields_dict = {}
        for field_ in fields:
            fields_dict[field_[xml_reader.ATT_NAME]] = field_
        self.assertEqual(fields_dict["data"][xml_reader.ATT_REFERENCE], "$0#")
        self.assertEqual(fields_dict["data_name"][xml_reader.ATT_VALUE], "Covariance matrix")
        self.assertEqual(fields_dict["label_x"][xml_reader.ATT_VALUE], "Nodes")
        #second output
        self.assertEqual(outputs[1][xml_reader.ATT_TYPE], "tvb.datatypes.arrays.MappedArray")
        fields = outputs[1][xml_reader.ELEM_FIELD]
        self.assertEqual(len(fields), 2)
        fields_dict = {}
        for field_ in fields:
            fields_dict[field_[xml_reader.ATT_NAME]] = field_
        self.assertEqual(fields_dict["data"][xml_reader.ATT_REFERENCE], "$0#")
        self.assertEqual(fields_dict["data_name"][xml_reader.ATT_VALUE], "Cross correlation")



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(XML_Reader_Test))
    return test_suite



if __name__ == "__main__":
    #To run tests individually.
    unittest.main()
    
    