# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import tvb.tests.framework.adapters.uploaders.test_data as test_data
import tvb_data.regionMapping as demo_data
import tvb_data.surfaceData
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.basic.neotraits.ex import TraitValueError
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neocom import h5
from tvb.core.services.exceptions import OperationException
from tvb.datatypes.surfaces import SurfaceTypesEnum
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestRegionMappingImporter(BaseTestCase):
    """
    Unit-tests for RegionMapping importer.
    """

    TXT_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'regionMapping_16k_76.txt')
    ZIP_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'regionMapping_16k_76.zip')
    BZ2_FILE = os.path.join(os.path.dirname(demo_data.__file__), 'regionMapping_16k_76.bz2')

    # Wrong data
    WRONG_FILE_1 = os.path.join(os.path.dirname(test_data.__file__), 'region_mapping_wrong_1.txt')
    WRONG_FILE_2 = os.path.join(os.path.dirname(test_data.__file__), 'region_mapping_wrong_2.txt')
    WRONG_FILE_3 = os.path.join(os.path.dirname(test_data.__file__), 'region_mapping_wrong_3.txt')

    def setup_method(self):
        """
        Sets up the environment for running the tests;
        creates a test user, a test project, a connectivity and a surface;
        imports a CFF data-set
        """
        self.test_user = TestFactory.create_user("UserRM")
        self.test_project = TestFactory.create_project(self.test_user)

        zip_path = os.path.join(os.path.dirname(tvb_data.__file__), 'connectivity', 'connectivity_76.zip')
        self.connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project, zip_path, "John")

        field = FilterChain.datatype + '.surface_type'
        filters = FilterChain('', [field], [SurfaceTypesEnum.CORTICAL_SURFACE.value], ['=='])
        cortex = os.path.join(os.path.dirname(tvb_data.surfaceData.__file__), 'cortex_16384.zip')
        TestFactory.import_surface_zip(self.test_user, self.test_project, cortex, SurfaceTypesEnum.CORTICAL_SURFACE)
        self.surface = TestFactory.get_entity(self.test_project, SurfaceIndex, filters)

    def teardown_method(self):
        """
        Clean-up tests data
        """
        self.clean_database()

    def test_import_no_surface_or_connectivity(self):
        """
        This method tests import of region mapping without providing a surface or connectivity
        """
        try:
            TestFactory.import_region_mapping(self.test_user, self.test_project, self.TXT_FILE,
                                              None, self.connectivity.gid, False)
            raise AssertionError("Import should fail in case Surface is missing")
        except TraitValueError:
            # Expected error
            pass

        try:
            TestFactory.import_region_mapping(self.test_user, self.test_project, self.TXT_FILE,
                                              self.surface.gid, None)
            raise AssertionError("Import should fail in case Connectivity is missing")
        except TraitValueError:
            # Expected error
            pass

    def test_import_from_txt(self):
        """
            This method tests import of region mapping from TXT file
        """
        self._import_from_file(self.TXT_FILE)

    def test_import_from_zip(self):
        """
            This method tests import of region mapping from TXT file
        """
        self._import_from_file(self.ZIP_FILE)

    def test_import_from_bz2(self):
        """
        This method tests import of region mapping from TXT file
        """
        self._import_from_file(self.BZ2_FILE)

    def _import_from_file(self, import_file):
        """
        This method tests import of region mapping from TXT file
        """
        region_mapping_index = TestFactory.import_region_mapping(self.test_user, self.test_project, import_file,
                                                                 self.surface.gid, self.connectivity.gid, False)

        surface_index = load_entity_by_gid(region_mapping_index.fk_surface_gid)
        assert surface_index is not None

        connectivity_index = load_entity_by_gid(region_mapping_index.fk_connectivity_gid)
        assert connectivity_index is not None

        region_mapping = h5.load_from_index(region_mapping_index)

        array_data = region_mapping.array_data
        assert array_data is not None
        assert 16384 == len(array_data)
        assert surface_index.number_of_vertices == len(array_data)

    def test_import_wrong_file_content(self):
        """
        This method tests import of region mapping with:
            - a wrong region number
            - wrong number of regions
            - negative region number
        """
        try:
            TestFactory.import_region_mapping(self.test_user, self.test_project, self.WRONG_FILE_1,
                                              self.surface.gid, self.connectivity.gid, False)
            raise AssertionError("Import should fail in case of invalid region number")
        except OperationException:
            # Expected exception
            pass

        try:
            # Execute some of these in the same process (to run faster, as they won't affect the logic)
            TestFactory.import_region_mapping(self.test_user, self.test_project, self.WRONG_FILE_2,
                                              self.surface.gid, self.connectivity.gid)
            raise AssertionError("Import should fail in case of invalid regions number")
        except (LaunchException, OperationException):
            # Expected exception
            pass

        try:
            TestFactory.import_region_mapping(self.test_user, self.test_project, self.WRONG_FILE_3,
                                              self.surface.gid, self.connectivity.gid)
            raise AssertionError("Import should fail in case of invalid region number (negative number)")
        except (LaunchException, OperationException):
            # Expected exception
            pass
