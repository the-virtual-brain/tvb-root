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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""
import os
import numpy
import unittest
import datetime
from tvb.core.utils import path2url_part, get_unique_file_name, string2date, date2string, string2bool
from tvb.core.utils import string2array
from tvb.tests.framework.core.base_testcase import TransactionalTestCase

class UtilsTest(TransactionalTestCase):
    """
    This class contains tests for the tvb.core.services.flowservice module.
    """  
    
    
    def setUp(self):
        """
        Reset the database before each test.
        """
        pass
    
    def tearDown(self):
        """
        Reset the database when test is done.
        """
        pass

    def test_path2url_part(self):
        """
        Test that all invalid characters are removed from the url.
        """
        processed_path = path2url_part("C:" + os.sep + "testtesttest test:aa")
        self.assertFalse(os.sep in processed_path, "Invalid character " + os.sep + " should have beed removed")
        self.assertFalse(' ' in processed_path, "Invalid character ' ' should have beed removed")
        self.assertFalse(':' in processed_path, "Invalid character ':' should have beed removed")
        
    def test_get_unique_file_name(self):
        """
        Test that we get unique file names no matter if we pass the same folder as input.
        """
        file_names = []
        nr_of_files = 100
        for _ in range(nr_of_files):
            file_name, _ = get_unique_file_name("", "file_name")
            fp = open(file_name, 'w')
            fp.write('test')
            fp.close()
            file_names.append(file_name)
        self.assertEqual(len(file_names), len(set(file_names)), 'No duplicate files should be generated.')
        for file_n in file_names:
            os.remove(file_n)
        
        
    def test_string2date(self):
        """
        Test the string2date function with different formats.
        """
        simple_time_string = "03-03-1999"
        simple_date = string2date(simple_time_string, complex_format=False)
        self.assertEqual(simple_date, datetime.datetime(1999, 3, 3), 
                         "Did not get expected datetime from conversion object.")
        
        complex_time_string = "1999-03-16,18-20-33.1"
        complex_date = string2date(complex_time_string)
        self.assertEqual(complex_date, datetime.datetime(1999, 3, 16, 18, 20, 33, 100000), 
                         "Did not get expected datetime from conversion object.")
        
        complex_time_stringv1 = "1999-03-16,18-20-33"
        complexv1_date = string2date(complex_time_stringv1)
        self.assertEqual(complexv1_date, datetime.datetime(1999, 3, 16, 18, 20, 33), 
                         "Did not get expected datetime from conversion object.")
        
        custom_format = "%Y"
        custom_time_string = "1999"
        custom_date = string2date(custom_time_string, date_format=custom_format)
        self.assertEqual(custom_date, datetime.datetime(1999, 1, 1), 
                         "Did not get expected datetime from conversion object.")
        
    def test_string2date_invalid(self):
        """
        Chech that a ValueError is raised in case some invalid date is passed.
        """
        self.assertRaises(ValueError, string2date, "somethinginvalid")
        
    def test_date2string(self):
        """
        Chech the date2string method for various inputs.
        """
        date_input = datetime.datetime(1999, 3, 16, 18, 20, 33, 100000)
        self.assertEqual(date2string(date_input, complex_format=False), '03-16-1999', 
                         "Did not get expected string from datetime conversion object.")
        
        custom_format = "%Y"
        self.assertEqual(date2string(date_input, date_format=custom_format), '1999', 
                         "Did not get expected string from datetime conversion object.")
        
        self.assertEqual(date2string(date_input, complex_format=True), '1999-03-16,18-20-33.100000', 
                         "Did not get expected string from datetime conversion object.")
        
        self.assertEqual("None", date2string(None), "Expected to return 'None' for None input.")
        
    def test_string2bool(self):
        """
        Chech the date2string method for various inputs.
        """
        self.assertTrue(string2bool("True"), "Expect True boolean for input 'True'")
        self.assertTrue(string2bool(u"True"), "Expect True boolean for input u'True'")
        self.assertTrue(string2bool("true"), "Expect True boolean for input 'true'")
        self.assertTrue(string2bool(u"true"), "Expect True boolean for input u'true'")
        self.assertFalse(string2bool("False"), "Expect True boolean for input 'False'")
        self.assertFalse(string2bool(u"False"), "Expect True boolean for input u'False'")
        self.assertFalse(string2bool("somethingelse"), "Expect True boolean for input 'somethingelse'")
        self.assertFalse(string2bool(u"somethingelse"), "Expect True boolean for input u'somethingelse'")
        
        
    def test_string2array(self):
        """
        Check the string2array method for various inputs
        """
        test_string_arrays = ['[1,2,3]', '1,2,3', '1 2 3']
        array_separators = [',', ',', ' ']
        for idx in xrange(len(test_string_arrays)):
            result_array = string2array(test_string_arrays[idx], array_separators[idx])
            self.assertEqual(len(result_array), 3)
            self.assertEqual(result_array[0], 1)
            self.assertEqual(result_array[1], 2)
            self.assertEqual(result_array[2], 3)
        
        
def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(UtilsTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)

