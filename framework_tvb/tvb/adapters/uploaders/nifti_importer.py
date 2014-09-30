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

from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.nifti.parser import NIFTIParser
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException, LaunchException
from tvb.core.entities.storage import transactional
from tvb.datatypes.time_series import TimeSeriesVolume
from tvb.datatypes.volumes import Volume


class NIFTIImporter(ABCUploader):
    """
    This importer is responsible for loading of data from NIFTI format (nii or nii.gz files)
    and store them in TVB as TimeSeries.
    """
    _ui_name = "NIFTI"
    _ui_subsection = "nifti_importer"
    _ui_description = "Import TimeSeries Volume from NIFTI"


    def get_upload_input_tree(self):
        """
            Take as input a GZ archive or NII file.
        """
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.nii, .gz, application/zip',
                 'label': 'Please select file to import (gz or nii)', 'required': True}]
        
        
    def get_output(self):
        return [Volume, TimeSeriesVolume]


    @transactional
    def launch(self, data_file):
        """
        Execute import operations:
        """
        parser = NIFTIParser(self.storage_path, self.operation_id)
        try:
            time_series = parser.parse(data_file)

            return [time_series.volume, time_series]             
        except ParseException, excep:
            logger = get_logger(__name__)
            logger.exception(excep)
            raise LaunchException(excep)