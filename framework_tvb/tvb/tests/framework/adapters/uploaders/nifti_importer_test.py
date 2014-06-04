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
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""
import unittest
import os
import numpy as numpy
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.tests.framework.datatypes.datatypes_factory import DatatypesFactory
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.datatypes.time_series import TimeSeries
from tvb.core.services.exceptions import OperationException

import tvb_data.nifti as demo_data



class NIFTIImporterTest(TransactionalTestCase):
    """
    Unit-tests for NIFTI importer.
    """

    NII_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'minimal.nii')
    GZ_NII_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'minimal.nii.gz')
    TVB_NII_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'avg152T1_LR_nifti.nii.gz')
    WRONG_NII_FILE = os.path.abspath(__file__)

    DEFAULT_ORIGIN = [[0.0, 0.0, 0.0]]
    UNKNOWN_STR = "unknown"


    def setUp(self):
        self.datatypeFactory = DatatypesFactory()
        self.test_project = self.datatypeFactory.get_project()
        self.test_user = self.datatypeFactory.get_user()


    def tearDown(self):
        """
        Clean-up tests data
        """
        FilesHelper().remove_project_structure(self.test_project.name)


    def _import(self, import_file_path=None):
        """
        This method is used for importing data in NIFIT format
        :param import_file_path: absolute path of the file to be imported
        """

        ### Retrieve Adapter instance 
        group = dao.find_group('tvb.adapters.uploaders.nifti_importer', 'NIFTIImporter')
        importer = ABCAdapter.build_adapter(group)
        args = {'data_file': import_file_path, DataTypeMetaData.KEY_SUBJECT: "bla bla"}

        ### Launch import Operation
        FlowService().fire_operation(importer, self.test_user, self.test_project.id, **args)

        time_series = TimeSeries()
        data_types = FlowService().get_available_datatypes(self.test_project.id,
                                                           time_series.module + "." + time_series.type)[0]
        self.assertEqual(1, len(data_types), "Project should contain only one data type.")

        time_series = ABCAdapter.load_entity_by_gid(data_types[0][2])
        self.assertTrue(time_series is not None, "TimeSeries should not be none")

        return time_series


    def test_import_demo_nii_data(self):
        """
            This method tests import of a NIFTI file.
        """
        time_series = self._import(self.TVB_NII_FILE)

        # Since self.assertAlmostEquals is not available on all machine
        # We compare floats as following
        self.assertTrue(abs(1.0 - time_series.sample_period) <= 0.001)
        self.assertEqual("sec", str(time_series.sample_period_unit))
        self.assertEqual(0.0, time_series.start_time)
        self.assertTrue(time_series.title.startswith("NIFTI"))

        data_shape = time_series.read_data_shape()
        self.assertEquals(4, len(data_shape))
        # We have only one entry for time dimension
        self.assertEqual(1, data_shape[0])
        dimension_labels = time_series.labels_ordering
        self.assertTrue(dimension_labels is not None)
        self.assertEquals(4, len(dimension_labels))

        volume = time_series.volume
        self.assertTrue(volume is not None)
        self.assertTrue(numpy.equal(self.DEFAULT_ORIGIN, volume.origin).all())
        self.assertEquals("mm", volume.voxel_unit)


    def test_import_nii_without_time_dimension(self):
        """
            This method tests import of a NIFTI file.
        """
        time_series = self._import(self.NII_FILE)

        self.assertEqual(1.0, time_series.sample_period)
        self.assertEqual(self.UNKNOWN_STR, str(time_series.sample_period_unit))
        self.assertEqual(0.0, time_series.start_time)
        self.assertTrue(time_series.title is not None)

        data_shape = time_series.read_data_shape()
        self.assertEquals(4, len(data_shape))
        # We have only one entry for time dimension
        self.assertEqual(1, data_shape[0])
        dimension_labels = time_series.labels_ordering
        self.assertTrue(dimension_labels is not None)
        self.assertEquals(4, len(dimension_labels))

        volume = time_series.volume
        self.assertTrue(volume is not None)
        self.assertTrue(numpy.equal(self.DEFAULT_ORIGIN, volume.origin).all())
        self.assertTrue(numpy.equal([3.0, 3.0, 3.0], volume.voxel_size).all())
        self.assertEquals(self.UNKNOWN_STR, volume.voxel_unit)


    def test_import_nifti_compressed(self):
        """
            This method tests import of a NIFTI file compressed in GZ format.
        """
        time_series = self._import(self.GZ_NII_FILE)

        self.assertEqual(1.0, time_series.sample_period)
        self.assertEqual(self.UNKNOWN_STR, str(time_series.sample_period_unit))
        self.assertEqual(0.0, time_series.start_time)
        self.assertTrue(time_series.title is not None)

        data_shape = time_series.read_data_shape()
        self.assertEquals(4, len(data_shape))
        # We have only one entry for time dimension
        self.assertEqual(1, data_shape[0])
        dimension_labels = time_series.labels_ordering
        self.assertTrue(dimension_labels is not None)
        self.assertEquals(4, len(dimension_labels))

        volume = time_series.volume
        self.assertTrue(volume is not None)
        self.assertTrue(numpy.equal(self.DEFAULT_ORIGIN, volume.origin).all())
        self.assertTrue(numpy.equal([3.0, 3.0, 3.0], volume.voxel_size).all())
        self.assertEquals(self.UNKNOWN_STR, volume.voxel_unit)


    def test_import_wrong_nii_file(self):
        """ 
        This method tests import of a file in a wrong format
        """
        try:
            self._import(self.WRONG_NII_FILE)
            self.fail("Import should fail in case of a wrong NIFTI format.")
        except OperationException:
            # Expected exception
            pass



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(NIFTIImporterTest, prefix="test_import_demo_nii_data"))
    return test_suite



if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)