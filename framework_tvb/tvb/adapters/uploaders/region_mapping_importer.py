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
import uuid
import numpy
import shutil
import zipfile
import tempfile
from tvb.datatypes.surfaces import CORTICAL
from tvb.adapters.uploaders.abcuploader import ABCUploader, ABCUploaderForm
from tvb.basic.filters.chain import FilterChain
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.datatypes.region_mapping_h5 import RegionMappingH5
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionMappingIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.neotraits._forms import UploadField, DataTypeSelectField
from tvb.core.neotraits.db import from_ndarray
from tvb.interfaces.neocom._h5loader import DirLoader


class RegionMappingImporterForm(ABCUploaderForm):

    def __init__(self, prefix='', project_id=None):
        super(RegionMappingImporterForm, self).__init__(prefix, project_id)

        self.mapping_file = UploadField('.txt, .zip, .bz2', self, name='mapping_file', required=True,
                                        label='Please upload region mapping file (txt, zip or bz2 format)',
                                        doc='Expected a text/zip/bz2 file containing region mapping values.')
        surface_conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=['=='],
                                         values=[CORTICAL])
        self.surface = DataTypeSelectField(SurfaceIndex, self, name='surface', required=True,
                                           conditions=surface_conditions, label='Brain Surface',
                                           doc='The Brain Surface used by uploaded region mapping.')
        self.connectivity = DataTypeSelectField(ConnectivityIndex, self, name='connectivity', label='Connectivity',
                                                required=True, doc='The Connectivity used by uploaded region mapping.')


class RegionMapping_Importer(ABCUploader):
    """
    Upload RegionMapping from a TXT, ZIP or BZ2 file.
    """
    _ui_name = "RegionMapping"
    _ui_subsection = "region_mapping_importer"
    _ui_description = "Import a Region Mapping (Surface - Connectivity) from TXT/ZIP/BZ2"

    logger = get_logger(__name__)

    form = None

    def get_input_tree(self): return None

    def get_upload_input_tree(self): return None

    def get_form(self):
        if self.form is None:
            return RegionMappingImporterForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_output(self):
        return [RegionMappingIndex]


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

        region_mapping_idx = RegionMappingIndex()
        region_mapping_idx.array_data_min, region_mapping_idx.array_data_max, region_mapping_idx.array_data_mean = from_ndarray(array_data)
        region_mapping_idx.surface = surface
        region_mapping_idx.surface_id = surface.id
        region_mapping_idx.connectivity = connectivity
        region_mapping_idx.connectivity_id = connectivity.id

        loader = DirLoader(self.storage_path)
        region_mapping_path = loader.path_for(RegionMappingH5, region_mapping_idx.gid)

        with RegionMappingH5(region_mapping_path) as region_mapping_h5:
            region_mapping_h5.array_data.store(array_data)
            region_mapping_h5.connectivity.store(uuid.UUID(connectivity.gid))
            region_mapping_h5.surface.store(uuid.UUID(surface.gid))
            region_mapping_h5.gid.store(uuid.UUID(region_mapping_idx.gid))

        return region_mapping_idx
