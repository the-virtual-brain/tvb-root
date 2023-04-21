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
.. moduleauthor:: Calin Pavel
"""

import tempfile
import zipfile

import numpy
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField, TraitDataTypeSelectField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces import Surface, SurfaceTypesEnum


class RegionMappingImporterModel(UploaderViewModel):
    mapping_file = Str(
        label='Please upload region mapping file (txt, zip or bz2 format)',
        doc='Expected a text/zip/bz2 file containing region mapping values.'
    )

    surface = DataTypeGidAttr(
        linked_datatype=Surface,
        label='Brain Surface',
        doc='The Brain Surface used by uploaded region mapping.'
    )

    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Connectivity',
        required=True, doc='The Connectivity used by uploaded region mapping.'
    )


class RegionMappingImporterForm(ABCUploaderForm):

    def __init__(self):
        super(RegionMappingImporterForm, self).__init__()

        self.mapping_file = TraitUploadField(RegionMappingImporterModel.mapping_file, ('.txt', '.zip', '.bz2'),
                                             'mapping_file')
        surface_conditions = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=['=='],
                                         values=[SurfaceTypesEnum.CORTICAL_SURFACE.value])
        self.surface = TraitDataTypeSelectField(RegionMappingImporterModel.surface, name='surface',
                                                conditions=surface_conditions)
        self.connectivity = TraitDataTypeSelectField(RegionMappingImporterModel.connectivity, name='connectivity')

    @staticmethod
    def get_view_model():
        return RegionMappingImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'mapping_file': ('.txt', '.zip', '.bz2')
        }


class RegionMappingImporter(ABCUploader):
    """
    Upload RegionMapping from a TXT, ZIP or BZ2 file.
    """
    _ui_name = "RegionMapping"
    _ui_subsection = "region_mapping_importer"
    _ui_description = "Import a Region Mapping (Surface - Connectivity) from TXT/ZIP/BZ2"

    logger = get_logger(__name__)

    def get_form_class(self):
        return RegionMappingImporterForm

    def get_output(self):
        return [RegionMappingIndex]

    def launch(self, view_model):
        # type: (RegionMappingImporterModel) -> [RegionMappingIndex]
        """
        Creates region mapping from uploaded data.
        :raises LaunchException: when
                    * a parameter is None or missing
                    * archive has more than one file
                    * uploaded files are empty
                    * number of vertices in imported file is different to the number of surface vertices
                    * imported file has negative values
                    * imported file has regions which are not in connectivity
        """
        if view_model.mapping_file is None:
            raise LaunchException("Please select mappings file which contains data to import")
        if view_model.surface is None:
            raise LaunchException("No surface selected. Please initiate upload again and select a brain surface.")
        if view_model.connectivity is None:
            raise LaunchException("No connectivity selected. Please initiate upload again and select one.")

        self.logger.debug("Reading mappings from uploaded file")

        if zipfile.is_zipfile(view_model.mapping_file):
            tmp_folder = tempfile.mkdtemp(prefix='region_mapping_zip_', dir=TvbProfile.current.TVB_TEMP_FOLDER)
            try:
                files = self.storage_interface.unpack_zip(view_model.mapping_file, tmp_folder)
                if len(files) > 1:
                    raise LaunchException("Please upload a ZIP file containing only one file.")
                array_data = self.read_list_data(files[0], dtype=numpy.int32)
            finally:
                self.storage_interface.remove_folder(tmp_folder, True)
        else:
            array_data = self.read_list_data(view_model.mapping_file, dtype=numpy.int32)

        # Now we do some checks before building final RegionMapping
        if array_data is None or len(array_data) == 0:
            raise LaunchException("Uploaded file does not contains any data. Please initiate upload with another file.")

        # Check if we have a mapping for each surface vertex.
        surface_idx = self.load_entity_by_gid(view_model.surface)
        if len(array_data) != surface_idx.number_of_vertices:
            msg = "Imported file contains a different number of values than the number of surface vertices. " \
                  "Imported: %d values while surface has: %d vertices." % (
                      len(array_data), surface_idx.number_of_vertices)
            raise LaunchException(msg)

        # Now check if the values from imported file correspond to connectivity regions
        if array_data.min() < 0:
            raise LaunchException("Imported file contains negative values. Please fix problem and re-import file")

        conn_idx = self.load_entity_by_gid(view_model.connectivity)
        if array_data.max() >= conn_idx.number_of_regions:
            msg = "Imported file contains invalid regions. Found region: %d while selected connectivity has: %d " \
                  "regions defined (0 based)." % (array_data.max(), conn_idx.number_of_regions)
            raise LaunchException(msg)

        self.logger.debug("Creating RegionMapping instance")

        surface_ht = h5.load_from_index(surface_idx)
        connectivity_ht = h5.load_from_index(conn_idx)

        region_mapping = RegionMapping(surface=surface_ht, connectivity=connectivity_ht, array_data=array_data)
        return self.store_complete(region_mapping)
