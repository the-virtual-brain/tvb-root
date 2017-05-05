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
.. moduleauthor:: Calin Pavel
"""

import os
import numpy
import shutil
import zipfile
import tempfile
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.connectivity import Connectivity



class RegionMapping_Importer(ABCUploader):
    """
    Upload RegionMapping from a TXT, ZIP or BZ2 file.
    """ 
    _ui_name = "RegionMapping"
    _ui_subsection = "region_mapping_importer"
    _ui_description = "Import a Region Mapping (Surface - Connectivity) from TXT/ZIP/BZ2"

    logger = get_logger(__name__)


    def get_upload_input_tree(self):
        """
        Define input parameters for this importer.
        """
        return [{'name': 'mapping_file', 'type': 'upload', 'required_type': '.txt, .zip, .bz2',
                 'label': 'Please upload region mapping file (txt, zip or bz2 format)', 'required': True,
                 'description': 'Expected a text/zip/bz2 file containing region mapping values.'},
                
                {'name': 'surface', 'label': 'Brain Surface', 
                 'type': CorticalSurface, 'required': True, 'datatype': True,
                 'description': 'The Brain Surface used by uploaded region mapping.'},
                
                {'name': 'connectivity', 'label': 'Connectivity', 
                 'type': Connectivity, 'required': True, 'datatype': True,
                 'description': 'The Connectivity used by uploaded region mapping.'}
                ]


    def get_output(self):
        return [RegionMapping]


    def launch(self, mapping_file, surface, connectivity):
        """
        Creates region mapping from uploaded data.

        :param mapping_file: an archive containing data for mapping surface to connectivity

        :raises LaunchException: when
                    * a parameter is None or missing
                    * archive has more than one file
                    * uploaded files are empty
                    * number of vertices in imported file is different to the number of surface vertices
                    * imported file has negative values
                    * imported file has regions which are not in connectivity
        """
        if mapping_file is None:
            raise LaunchException("Please select mappings file which contains data to import")
        if surface is None:
            raise LaunchException("No surface selected. Please initiate upload again and select a brain surface.")
        if connectivity is None:
            raise LaunchException("No connectivity selected. Please initiate upload again and select one.")
            
        self.logger.debug("Reading mappings from uploaded file")

        if zipfile.is_zipfile(mapping_file):
            tmp_folder = tempfile.mkdtemp(prefix='region_mapping_zip_', dir=TvbProfile.current.TVB_TEMP_FOLDER)
            try:
                files = FilesHelper().unpack_zip(mapping_file, tmp_folder)
                if len(files) > 1:
                    raise LaunchException("Please upload a ZIP file containing only one file.")
                array_data = self.read_list_data(files[0], dtype=numpy.int32)
            finally:
                if os.path.exists(tmp_folder):
                    shutil.rmtree(tmp_folder)
        else:
            array_data = self.read_list_data(mapping_file, dtype=numpy.int32)
        
        # Now we do some checks before building final RegionMapping
        if array_data is None or len(array_data) == 0:
            raise LaunchException("Uploaded file does not contains any data. Please initiate upload with another file.")
        
        # Check if we have a mapping for each surface vertex.
        if len(array_data) != surface.number_of_vertices:
            msg = "Imported file contains a different number of values than the number of surface vertices. " \
                  "Imported: %d values while surface has: %d vertices." % (len(array_data), surface.number_of_vertices)
            raise LaunchException(msg)     
        
        # Now check if the values from imported file correspond to connectivity regions
        if array_data.min() < 0:
            raise LaunchException("Imported file contains negative values. Please fix problem and re-import file")
        
        if array_data.max() >= connectivity.number_of_regions:
            msg = "Imported file contains invalid regions. Found region: %d while selected connectivity has: %d " \
                  "regions defined (0 based)." % (array_data.max(), connectivity.number_of_regions)
            raise LaunchException(msg)
        
        self.logger.debug("Creating RegionMapping instance")
        region_mapping_inst = RegionMapping()
        region_mapping_inst.storage_path = self.storage_path
        region_mapping_inst.surface = surface
        region_mapping_inst.connectivity = connectivity
        
        if array_data is not None:
            region_mapping_inst.array_data = array_data
        
        return [region_mapping_inst]
    
    