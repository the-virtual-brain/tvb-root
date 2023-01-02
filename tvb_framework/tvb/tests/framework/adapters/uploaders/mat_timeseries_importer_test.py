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
Unit-test for mat_timeseries_importer and mat_parser.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os

import tvb_data
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.uploaders.mat_timeseries_importer import RegionMatTimeSeriesImporterModel, RegionTimeSeriesImporter
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


class TestMatTimeSeriesImporter(BaseTestCase):
    base_pth = os.path.join(os.path.dirname(tvb_data.__file__), 'berlinSubjects', 'QL_20120814')
    bold_path = os.path.join(base_pth, 'QL_BOLD_regiontimecourse.mat')
    connectivity_path = os.path.join(base_pth, 'QL_20120814_Connectivity.zip')

    def setup_method(self):
        self.test_user = TestFactory.create_user('Mat_Timeseries_User')
        self.test_project = TestFactory.create_project(self.test_user, "Mat_Timeseries_Project")
        self.connectivity = TestFactory.import_zip_connectivity(self.test_user, self.test_project,
                                                                self.connectivity_path)

    def teardown_method(self):
        self.clean_database()

    def test_import_bold(self):
        view_model = RegionMatTimeSeriesImporterModel()
        view_model.data_file = self.bold_path
        view_model.dataset_name = "QL_20120824_DK_BOLD_timecourse"
        view_model.data_subject = "QL"
        view_model.datatype = self.connectivity.gid

        TestFactory.launch_importer(RegionTimeSeriesImporter, view_model, self.test_user, self.test_project, False)

        tsr = TestFactory.get_entity(self.test_project, TimeSeriesRegionIndex)
        assert (661, 1, 68, 1) == (tsr.data_length_1d, tsr.data_length_2d, tsr.data_length_3d, tsr.data_length_4d)
