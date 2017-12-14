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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.datatypes.surfaces import CorticalSurface
from tvb.core.services.exceptions import OperationException
from tvb.adapters.uploaders.gifti.parser import GIFTIParser

import tvb_data.gifti as demo_data



class TestGIFTISurfaceImporter(TransactionalTestCase):
    """
    Unit-tests for GIFTI Surface importer.
    """

    GIFTI_SURFACE_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'sample.cortex.gii')
    GIFTI_TIME_SERIES_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'sample.time_series.gii')
    WRONG_GII_FILE = os.path.abspath(__file__)


    def transactional_setup_method(self):
        self.datatypeFactory = DatatypesFactory()
        self.test_project = self.datatypeFactory.get_project()
        self.test_user = self.datatypeFactory.get_user()


    def transactional_teardown_method(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)


    def _importSurface(self, import_file_path=None):
        """
        This method is used for importing data in GIFIT format
        :param import_file_path: absolute path of the file to be imported
        """

        ### Retrieve Adapter instance 
        importer = TestFactory.create_adapter('tvb.adapters.uploaders.gifti_surface_importer', 'GIFTISurfaceImporter')

        args = {'data_file': import_file_path, DataTypeMetaData.KEY_SUBJECT: ""}

        ### Launch import Operation
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)

        surface = CorticalSurface()
        data_types = FlowService().get_available_datatypes(self.test_project.id,
                                                           surface.module + "." + surface.type)[0]
        assert 1, len(data_types) == "Project should contain only one data type."

        surface = ABCAdapter.load_entity_by_gid(data_types[0][2])
        assert surface is not None == "TimeSeries should not be none"

        return surface


    def test_import_surface_gifti_data(self):
        """
            This method tests import of a surface from GIFTI file.
            !!! Important: We changed this test to execute only GIFTI parse
                because storing surface it takes too long (~ 9min) since
                normals needs to be calculated.
        """
        operation_id = self.datatypeFactory.get_operation().id
        storage_path = FilesHelper().get_operation_folder(self.test_project.name, operation_id)

        parser = GIFTIParser(storage_path, operation_id)
        surface = parser.parse(self.GIFTI_SURFACE_FILE)

        assert 131342 == len(surface.vertices)
        assert 262680 == len(surface.triangles)


    def test_import_timeseries_gifti_data(self):
        """
        This method tests import of a time series from GIFTI file.
        !!! Important: We changed this test to execute only GIFTI parse
            because storing surface it takes too long (~ 9min) since
            normals needs to be calculated.
        """
        operation_id = self.datatypeFactory.get_operation().id
        storage_path = FilesHelper().get_operation_folder(self.test_project.name, operation_id)

        parser = GIFTIParser(storage_path, operation_id)
        time_series = parser.parse(self.GIFTI_TIME_SERIES_FILE)

        data_shape = time_series.read_data_shape()

        assert 135 == data_shape[0]
        assert 143479 == data_shape[1]


    def test_import_wrong_gii_file(self):
        """ 
        This method tests import of a file in a wrong format
        """
        try:
            self._importSurface(self.WRONG_GII_FILE)
            raise AssertionError("Import should fail in case of a wrong GIFTI format.")
        except OperationException:
            # Expected exception
            pass

