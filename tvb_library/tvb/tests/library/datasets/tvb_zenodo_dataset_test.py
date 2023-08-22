
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
import zipfile
import pytest

class Test_TVBZenodoDataset(BaseTestCase):
     
    
    def test_extract(self):

        dataset = TVBZenodoDataset()
        connectivity66_dir = Path(dataset.fetch_data("connectivity_66.zip"))

        assert str(connectivity66_dir).endswith(".zip")
        assert connectivity66_dir.is_file()
        dataset.delete_data()
        assert not connectivity66_dir.is_file() 

        dataset = TVBZenodoDataset(version="2.0.3", extract_dir="dataset")
        connectivity66_dir = Path(dataset.fetch_data("connectivity_66.zip"))

        assert str(connectivity66_dir).endswith(".zip")
        assert "dataset" in str(connectivity66_dir)
        assert (Path.cwd()/"dataset").is_dir()
        assert (Path.cwd()/"dataset"/"tvb_data").is_dir()
        dataset.delete_data()
        assert not connectivity66_dir.is_file() 

        dataset =  TVBZenodoDataset(version="2.0.3", extract_dir="~/dataset") 
        matfile_dir = Path(dataset.fetch_data("local_connectivity_80k.mat"))
        
        assert str(matfile_dir).endswith(".mat")
        assert matfile_dir.is_file()
        dataset.delete_data()
        assert not matfile_dir.is_file()

        
        excel_extract = Path(dataset.fetch_data(" ConnectivityTable_regions.xls"))
        assert excel_extract.is_file()
        dataset.delete_data()
        assert not excel_extract.is_file()


    def test_check_content(self):

        #check if connectivity_66 contains expected files.
        dataset = TVBZenodoDataset()
        connectivity66_dir = Path(dataset.fetch_data("connectivity_66.zip"))

        assert "centres.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        assert "info.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        assert "tract_lengths.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        assert "weights.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        dataset.delete_data()

        
        dataset = TVBZenodoDataset(version= "2.0.3", extract_dir="~/dataset")
        connectivity66_dir = Path(dataset.fetch_data("connectivity_66.zip"))
        assert "centres.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        assert "info.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        assert "tract_lengths.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        assert "weights.txt" in zipfile.ZipFile(connectivity66_dir).namelist()
        dataset.delete_data()


    def test_file_name_variants(self):
        dataset = TVBZenodoDataset(version= "2.0.3", extract_dir="~/dataset")
        connectivity66_dir_1 = Path(dataset.fetch_data("connectivity_66.zip"))
        connectivity66_dir_2 = Path(dataset.fetch_data('tvb_data/connectivity/connectivity_66.zip'))
        assert connectivity66_dir_1 == connectivity66_dir_2

        dataset.delete_data()

        dataset = TVBZenodoDataset()
        connectivity66_dir_1 = Path(dataset.fetch_data("connectivity_66.zip"))
        connectivity66_dir_2 = Path(dataset.fetch_data('tvb_data/connectivity/connectivity_66.zip'))
        assert connectivity66_dir_1 == connectivity66_dir_2

        dataset.delete_data()

        
        dataset = TVBZenodoDataset(version= "2.0.3", extract_dir="dataset")
        connectivity66_dir_1 = Path(dataset.fetch_data("connectivity_66.zip"))
        connectivity66_dir_2 = Path(dataset.fetch_data('tvb_data/connectivity/connectivity_66.zip'))
        assert connectivity66_dir_1 == connectivity66_dir_2

        dataset.delete_data()


        # should raise error cause there are two files with name mapping_FS_84.txt
        with pytest.raises(NameError):
            dataset = TVBZenodoDataset()
            data = dataset.fetch_data("mapping_FS_84.txt")
        
        # no error when relative path given
        dataset = TVBZenodoDataset()
        data = Path(dataset.fetch_data(" tvb_data/macaque/mapping_FS_84.txt"))
        assert data.is_file()
        
        data = Path(dataset.fetch_data('tvb_data/nifti/volume_mapping/mapping_FS_84.txt'))
        assert data.is_file()

        dataset.delete_data()
