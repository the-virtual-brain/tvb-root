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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import os
import pytest
from datetime import datetime
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.utils import path2url_part, get_unique_file_name, string2date, date2string, string2bool


class TestUtils(TransactionalTestCase):
    """
    This class contains test helper methods.
    """

    def test_path2url_part(self):
        """
        Test that all invalid characters are removed from the url.
        """
        processed_path = path2url_part("C:" + os.sep + "testtesttest test:aa")
        assert not os.sep in processed_path, "Invalid character " + os.sep + " should have beed removed"
        assert not ' ' in processed_path, "Invalid character ' ' should have beed removed"
        assert not ':' in processed_path, "Invalid character ':' should have beed removed"

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
        assert len(file_names) == len(set(file_names)), 'No duplicate files should be generated.'
        for file_n in file_names:
            os.remove(file_n)

    def test_string2date(self):
        """
        Test the string2date function with different formats.
        """
        simple_time_string = "03-03-1999"
        simple_date = string2date(simple_time_string, complex_format=False)
        assert simple_date == datetime(1999, 3, 3),\
                         "Did not get expected datetime from conversion object."
        complex_time_string = "1999-03-16,18-20-33.1"
        complex_date = string2date(complex_time_string)
        assert complex_date == datetime(1999, 3, 16, 18, 20, 33, 100000),\
                         "Did not get expected datetime from conversion object."
        complex_time_stringv1 = "1999-03-16,18-20-33"
        complexv1_date = string2date(complex_time_stringv1)
        assert complexv1_date == datetime(1999, 3, 16, 18, 20, 33),\
                         "Did not get expected datetime from conversion object."
        custom_format = "%Y"
        custom_time_string = "1999"
        custom_date = string2date(custom_time_string, date_format=custom_format)
        assert custom_date == datetime(1999, 1, 1),\
                         "Did not get expected datetime from conversion object."

    def test_string2date_invalid(self):
        """
        Check that a ValueError is raised in case some invalid date is passed.
        """
        with pytest.raises(ValueError):
            string2date("somethinginvalid")

    def test_date2string(self):
        """
        Check the date2string method for various inputs.
        """
        date_input = datetime(1999, 3, 16, 18, 20, 33, 100000)
        assert date2string(date_input, complex_format=False) == '03-16-1999',\
                         "Did not get expected string from datetime conversion object."
        custom_format = "%Y"
        assert date2string(date_input, date_format=custom_format) == '1999',\
                         "Did not get expected string from datetime conversion object."

        assert date2string(date_input, complex_format=True) == '1999-03-16,18-20-33.100000',\
                         "Did not get expected string from datetime conversion object."

        assert "None" == date2string(None), "Expected to return 'None' for None input."

    def test_string2bool(self):
        """
        Check the date2string method for various inputs.
        """
        assert string2bool("True"), "Expect True boolean for input 'True'"
        assert string2bool("True"), "Expect True boolean for input u'True'"
        assert string2bool("true"), "Expect True boolean for input 'true'"
        assert string2bool("true"), "Expect True boolean for input u'true'"
        assert not string2bool("False"), "Expect True boolean for input 'False'"
        assert not string2bool("False"), "Expect True boolean for input u'False'"
        assert not string2bool("somethingelse"), "Expect True boolean for input 'somethingelse'"
        assert not string2bool("somethingelse"), "Expect True boolean for input u'somethingelse'"
