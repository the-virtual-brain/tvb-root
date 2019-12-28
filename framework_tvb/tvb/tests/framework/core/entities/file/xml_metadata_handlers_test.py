# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import json
from tvb.core.entities.model.model_operation import STATUS_FINISHED
from tvb.core.entities.transient.structure_entities import GenericMetaData
from tvb.core.entities.file.xml_metadata_handlers import XMLReader, XMLWriter


class TestMetaDataReadXML():
    """
    Tests for tvb.core.entities.file.metadatahandler.XMLReader class.
    """   
    TO_BE_READ_FILE = "test_read.xml"
    #Values expected to be read from file
    EXPECTED_DICTIONARY = {'status': STATUS_FINISHED,
                           'gid': '497b3d59-b3c1-11e1-b2e4-68a86d1bd4fa',
                           'user_group': 'cff_74',
                           'fk_from_algo': json.dumps({'classname': 'CFF_Importer', 'identifier': None,
                                                       'module': 'tvb.adapters.uploaders.cff_importer'})
                           }
        
    def setup_method(self):
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
        assert isinstance(meta_data, GenericMetaData)
        for key, value in self.EXPECTED_DICTIONARY.items():
            found_value = meta_data[key]
            assert value == found_value
        
    def test_read_gid(self):
        """
        Test that value returned by read_only_element matches the actual value from the XML file.
        """
        read_value = self.meta_reader.read_only_element('gid')
        assert isinstance(read_value, str)
        assert read_value == self.EXPECTED_DICTIONARY['gid']

     
class TestMetaDataWriteXML():
    """
    Tests for XMLWriter.
    """ 
    WRITABLE_METADATA = TestMetaDataReadXML.EXPECTED_DICTIONARY
    
    def setup_method(self):
        """
        Sets up necessary files for the tests.
        """
        meta_data_entity = GenericMetaData(self.WRITABLE_METADATA)
        self.meta_writer = XMLWriter(meta_data_entity)
        self.result_path = os.path.join(os.path.dirname(__file__), "Operation.xml")
      
    def teardown_method(self):
        """
        Remove created XML file.
        """
        os.remove(self.result_path)
        
    def test_write_metadata(self):
        """
        Test that an XML file is created and correct data is written in it.
        """
        assert not os.path.exists(self.result_path)
        self.meta_writer.write(self.result_path)
        assert os.path.exists(self.result_path)
        reader = XMLReader(self.result_path)
        meta_data = reader.read_metadata()
        for key, value in TestMetaDataReadXML.EXPECTED_DICTIONARY.items():
            found_value = meta_data[key]
            assert value == found_value
