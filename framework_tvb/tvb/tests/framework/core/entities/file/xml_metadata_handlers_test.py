# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
"""

import os
import unittest
import json
from tvb.core.entities import model
from tvb.core.entities.transient.structure_entities import GenericMetaData
from tvb.core.entities.file.xml_metadata_handlers import XMLReader, XMLWriter




class MetaDataReadXMLTest(unittest.TestCase):
    """
    Tests for tvb.core.entities.file.metadatahandler.XMLReader class.
    """   
    TO_BE_READ_FILE = "test_read.xml"
    #Values expected to be read from file
    EXPECTED_DICTIONARY = {'status': model.STATUS_FINISHED,
                           'gid': '497b3d59-b3c1-11e1-b2e4-68a86d1bd4fa',
                           'user_group': 'cff_74',
                           'fk_from_algo': json.dumps({'classname': 'CFF_Importer', 'identifier': None,
                                                       'module': 'tvb.adapters.uploaders.cff_importer'})
                           }
        
    def setUp(self):
        """
        Sets up necessary files for the tests.
        """
        self.file_path = os.path.join(os.path.dirname(__file__), 
                                      self.TO_BE_READ_FILE)
        self.meta_reader = XMLReader(self.file_path)
    
    def test_read_metadata(self):
        """
        Test that content return by read_metadata matches the
        actual content of the XML.
        """
        meta_data = self.meta_reader.read_metadata()
        self.assertTrue(isinstance(meta_data, GenericMetaData))
        for key, value in self.EXPECTED_DICTIONARY.iteritems():
            found_value = meta_data[key]
            self.assertEqual(value, found_value)
        
    def test_read_gid(self):
        """
        Test that value returned by read_only_element matches the actual value from the XML file.
        """
        read_value = self.meta_reader.read_only_element('gid')
        self.assertTrue(isinstance(read_value, str))
        self.assertEqual(read_value, self.EXPECTED_DICTIONARY['gid'])
        
        
        
     
class MetaDataWriteXMLTest(unittest.TestCase):  
    """
    Tests for XMLWriter.
    """ 
    WRITABLE_METADATA = MetaDataReadXMLTest.EXPECTED_DICTIONARY
    
    def setUp(self):
        """
        Sets up necessary files for the tests.
        """
        meta_data_entity = GenericMetaData(self.WRITABLE_METADATA)
        self.meta_writer = XMLWriter(meta_data_entity)
        self.result_path = os.path.join(os.path.dirname(__file__), "Operation.xml")
      
    def tearDown(self):
        """
        Remove created XML file.
        """
        unittest.TestCase.tearDown(self)
        os.remove(self.result_path)
        
    def test_write_metadata(self):
        """
        Test that an XML file is created and correct data is written in it.
        """
        self.assertFalse(os.path.exists(self.result_path))
        self.meta_writer.write(self.result_path)
        self.assertTrue(os.path.exists(self.result_path))
        reader = XMLReader(self.result_path)
        meta_data = reader.read_metadata()
        for key, value in MetaDataReadXMLTest.EXPECTED_DICTIONARY.iteritems():
            found_value = meta_data[key]
            self.assertEqual(value, found_value)
    
    

def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(MetaDataReadXMLTest))
    test_suite.addTest(unittest.makeSuite(MetaDataWriteXMLTest))
    return test_suite


if __name__ == "__main__":
    unittest.main()
    
    
    
