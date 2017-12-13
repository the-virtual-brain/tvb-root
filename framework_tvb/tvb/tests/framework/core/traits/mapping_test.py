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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import numpy
import copy
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.datatypes.arrays import MappedArray
from tvb.basic.traits import types_basic as basic
from tvb.basic.traits.types_mapped import MappedType
from tvb.core.entities import model
from tvb.core.entities.storage import dao, SA_SESSIONMAKER
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.core.factory import TestFactory


class MappedTestClass(MappedType):
    """Simple traited datatype for tests"""
    dikt = basic.Dict
    tup = basic.Tuple
    dtype = basic.DType
    json = basic.JSONType
  

class TestMapping(BaseTestCase):
    """
    This class contains tests for the tvb.core.datatype module.
    """  
    
    def setup_method(self):
        """
        Reset the database before each test.
        """
        self.clean_database()
        self.flow_service = FlowService()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(admin=self.test_user)
        self.operation = TestFactory.create_operation(test_user=self.test_user, test_project=self.test_project)

    
    def teardown_method(self):
        """
        Reset the database when test is done.
        """
        self.clean_database()
    
     
    def test_db_mapping(self):
        """ Test DB storage/retrieval of a simple traited attribute"""
        session = SA_SESSIONMAKER()
        model.Base.metadata.create_all(bind=session.connection())
        session.commit()
        session.close()
        
        # test data
        dikt = {'a': 6}
        tup = ('5', 9.348)
        dtype = numpy.dtype(float)
        json = {'a': 'asdf', 'b': {'23': '687568'}}

        test_inst = MappedTestClass()
        test_inst.dikt = copy.deepcopy(dikt)
        test_inst.tup = copy.deepcopy(tup)
        test_inst.dtype = copy.deepcopy(dtype)
        test_inst.json = copy.deepcopy(json)
        test_inst.set_operation_id(self.operation.id)
        test_inst = dao.store_entity(test_inst)

        test_inst = dao.get_generic_entity(MappedTestClass, test_inst.gid, 'gid')[0]
        assert  test_inst.dikt == dikt
        assert  test_inst.tup == tup
        assert  test_inst.dtype == dtype
        assert  test_inst.json == json

    
        
    def test_read_write_arrays(self):
        """
        Test the filter function when retrieving dataTypes with a filter
        after a column from a class specific table (e.g. DATA_arraywrapper).
        """ 
        test_array = numpy.array(range(16))
        shapes = [test_array.shape, (2, 8), (2, 2, 4), (2, 2, 2, 2)]
        storage_path = self.flow_service.file_helper.get_project_folder(self.operation.project, str(self.operation.id))
        for i in range(4):
            datatype_inst = MappedArray(title="dim_" + str(i + 1), d_type="MappedArray",
                                        storage_path=storage_path, module="tvb.datatypes.arrays",
                                        subject="John Doe", state="RAW", operation_id=self.operation.id)
            datatype_inst.array_data = test_array.reshape(shapes[i])
            result = dao.store_entity(datatype_inst)
            result.array_data = None
        
        inserted_data = self.flow_service.get_available_datatypes(self.test_project.id,
                                                                  "tvb.datatypes.arrays.MappedArray")[0]
        assert  len(inserted_data) == 4, "Found " + str(len(inserted_data))
 
        for i in range(4):
            ## inserted_data will be retrieved in the opposite order than the insert order
            actual_datatype = dao.get_generic_entity(MappedArray, inserted_data[3 - i][2], 'gid')[0]
            assert  actual_datatype.length_1d, shapes[i][0]
            if i > 0:
                assert  actual_datatype.length_2d == shapes[i][1]
            expected_arr = test_array.reshape(shapes[i])
            assert numpy.equal(actual_datatype.array_data, expected_arr).all(),\
                            str(i + 1) + "D Data not read correctly"
            actual_datatype.array_data = None
            ### Check that meta-data are also written for Array attributes.
            metadata = actual_datatype.get_metadata('array_data')
            assert actual_datatype.METADATA_ARRAY_MAX in metadata
            assert  metadata[actual_datatype.METADATA_ARRAY_MAX] == 15
            assert actual_datatype.METADATA_ARRAY_MIN in metadata
            assert  metadata[actual_datatype.METADATA_ARRAY_MIN] == 0
            assert actual_datatype.METADATA_ARRAY_MEAN in metadata
            assert  metadata[actual_datatype.METADATA_ARRAY_MEAN] == 7.5
        