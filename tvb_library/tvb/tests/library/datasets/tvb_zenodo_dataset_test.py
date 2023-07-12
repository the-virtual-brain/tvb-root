
# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
.. moduleauthor:: Abhijit Deo <f20190041@goa.bits-pilani.ac.in>
"""

from tvb.datasets import TVBZenodoDataset
from pathlib import Path
from tvb.tests.library.base_testcase import BaseTestCase

class Test_TVBZenodoDataset(BaseTestCase):
     
    
    def test_extract(self):

        tvb_data = TVBZenodoDataset()
        connectivity66_dir = Path(tvb_data.fetch_data("connectivity_66.zip"))
        assert connectivity66_dir.is_file()
        tvb_data.delete_data()
        assert not connectivity66_dir.is_file() 

        tvb_data = TVBZenodoDataset(version="2.0.3", extract_dir="tvb_data")
        connectivity66_dir = Path(tvb_data.fetch_data("connectivity_66.zip"))
        assert connectivity66_dir.is_file()
        tvb_data.delete_data()
        assert not connectivity66_dir.is_file() 

        tvb_data =  TVBZenodoDataset(version="2.0.3", extract_dir="~/tvb_data") 
        matfile_dir = Path(tvb_data.fetch_data("local_connectivity_80k.mat"))
        assert matfile_dir.is_file()
        tvb_data.delete_data()
        assert not matfile_dir.is_file()


        all_extract = Path(TVBZenodoDataset(version = "2.0.3", extract_dir="~/tvb_data").fetch_data(" ConnectivityTable_regions.xls"))
        assert all_extract.is_file()
        tvb_data.delete_data()
        assert not all_extract.is_file()

    #TODO add no interenet tests
