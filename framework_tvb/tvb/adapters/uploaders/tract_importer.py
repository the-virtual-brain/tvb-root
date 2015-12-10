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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
from nibabel import trackvis
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.files_helper import TvbZip
from tvb.core.entities.storage import transactional
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.tracts import Tracts


def chunk_iter(iterable, n):
    """
    Reads a generator in chunks. Yields lists. Last one may be smaller than n.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk



class _TrackImporterBase(ABCUploader):
    _ui_name = "Tracts"
    _ui_subsection = "tracts_importer"
    _ui_description = "Import tracts"

    READ_CHUNK = 4*1024

    def get_upload_input_tree(self):
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.trk',
                 'label': 'Please select file to import', 'required': True},
                {'name': 'region_volume', 'type': RegionVolumeMapping,
                 'label': 'Reference Volume Map', 'required': False}]


    def get_output(self):
        return [Tracts]


    def _get_tract_region(self, region_volume, start_vertex):
        # Map to voxel index space
        # Lacking any affine matrix between these, we assume they are in the same geometric space
        # What remains is to map geometry to the discrete region volume mapping indices
        x_plane, y_plane, z_plane = [int(i) for i in start_vertex]
        rvshape = region_volume.read_data_shape()

        if not (0 <= x_plane < rvshape[0] and 0 <= y_plane < rvshape[1] or 0 <= z_plane < rvshape[2]):
            raise IndexError('vertices outside the region volume map cube')

        # in memory data set
        if self.full_rmap_cache is not None:
            region_id = self.full_rmap_cache[x_plane, y_plane, z_plane]
            return region_id

        # not in memory have to go to disk
        slices = slice(x_plane, x_plane + 1), slice(y_plane, y_plane + 1), slice(z_plane, z_plane + 1)
        region_id = region_volume.read_data_slice(slices)[0, 0, 0]
        return region_id


    def _attempt_to_cache_regionmap(self, region_volume):
        a, b, c = region_volume.read_data_shape()
        if a*b*c <= 256*256*256:
            # read all
            slices = slice(a), slice(b), slice(c)
            self.full_rmap_cache = region_volume.read_data_slice(slices)
        else:
            self.full_rmap_cache = None



class TrackvizTractsImporter(_TrackImporterBase):
    """
    This imports tracts from the trackviz format
    """

    @transactional
    def launch(self, data_file, region_volume=None):
        if data_file is None:
            raise LaunchException("Please select a trackviz file")

        if region_volume is not None:
            self._attempt_to_cache_regionmap(region_volume)

        datatype = Tracts()
        datatype.storage_path = self.storage_path

        tract_gen, hdr = trackvis.read(data_file, as_generator=True)

        tract_start_indices = [0]
        tract_region = []

        for tract_bundle in chunk_iter(tract_gen, self.READ_CHUNK):
            tract_bundle = [tr[0] for tr in tract_bundle]

            for tr in tract_bundle:
                tract_start_indices.append(tract_start_indices[-1] + len(tr))
                if region_volume is not None:
                    tract_region.append(self._get_tract_region(region_volume, tr[0]))

            vertices = numpy.concatenate(tract_bundle)
            datatype.store_data_chunk("vertices", vertices, grow_dimension=0, close_file=False)

        datatype.tract_start_idx = tract_start_indices
        datatype.tract_region = numpy.array(tract_region, dtype=numpy.int16)
        return datatype



class ZipTxtTractsImporter(_TrackImporterBase):
    """
    This imports tracts from a zip containing txt files. One txt file for a tract.
    """
    _ui_name = "ZipTxtTracts"


    def get_upload_input_tree(self):
        tree = _TrackImporterBase.get_upload_input_tree(self)
        tree[0]['required_type'] = '.zip'
        return tree


    @transactional
    def launch(self, data_file, region_volume=None):
        if data_file is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        if region_volume is not None:
            self._attempt_to_cache_regionmap(region_volume)

        datatype = Tracts()
        datatype.storage_path = self.storage_path

        tract_start_indices = [0]
        tract_region = []

        with TvbZip(data_file) as zipf:
            for tractf in sorted(zipf.namelist()): # one track per file
                if not tractf.endswith('.txt'): # omit directories and other non track files
                    continue
                vertices_file = zipf.open(tractf)
                tract_vertices = numpy.loadtxt(vertices_file, dtype=numpy.float32)

                tract_start_indices.append(tract_start_indices[-1] + len(tract_vertices))
                datatype.store_data_chunk("vertices", tract_vertices, grow_dimension=0, close_file=False)

                if region_volume is not None:
                    tract_region.append(self._get_tract_region(region_volume, tract_vertices[0]))
                vertices_file.close()

        datatype.tract_start_idx = tract_start_indices
        datatype.tract_region = numpy.array(tract_region, dtype=numpy.int16)
        return datatype
