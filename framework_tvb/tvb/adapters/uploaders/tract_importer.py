# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
from abc import ABCMeta
from nibabel import trackvis
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.files_helper import TvbZip
from tvb.adapters.datatypes.db.tracts import TractsIndex
from tvb.core.entities.storage import transactional
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.region_mapping import RegionVolumeMapping
from tvb.datatypes.tracts import Tracts
from tvb.core.neotraits.forms import TraitUploadField, TraitDataTypeSelectField


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


class TrackImporterModel(UploaderViewModel):
    data_file = Str(
        label='Please select file to import'
    )

    region_volume = DataTypeGidAttr(
        linked_datatype=RegionVolumeMapping,
        required=False,
        label='Reference Volume Map'
    )


class TrackImporterForm(ABCUploaderForm):

    def __init__(self, project_id=None):
        super(TrackImporterForm, self).__init__(project_id)

        self.data_file = TraitUploadField(TrackImporterModel.data_file, '.trk', self.project_id, 'data_file',
                                          self.temporary_files)
        self.region_volume = TraitDataTypeSelectField(TrackImporterModel.region_volume, self.project_id,
                                                      name='region_volume')

    @staticmethod
    def get_view_model():
        return TrackImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.trk'
        }


class TrackZipImporterForm(TrackImporterForm):

    def __init__(self, project_id=None):
        super(TrackZipImporterForm, self).__init__(project_id)

        self.data_file.required_type = '.zip'


class _TrackImporterBase(ABCUploader, metaclass=ABCMeta):
    _ui_name = "Tracts TRK"
    _ui_subsection = "tracts_importer"
    _ui_description = "Import tracts"

    READ_CHUNK = 4 * 1024

    def get_form_class(self):
        return TrackImporterForm

    def get_output(self):
        return [TractsIndex]

    def _get_tract_region(self, start_vertex):
        # Map to voxel index space
        # Lacking any affine matrix between these, we assume they are in the same geometric space
        # What remains is to map geometry to the discrete region volume mapping indices
        x_plane, y_plane, z_plane = [int(i) for i in start_vertex]

        if not (0 <= x_plane < self.region_volume_shape[0] and
                0 <= y_plane < self.region_volume_shape[1] and
                0 <= z_plane < self.region_volume_shape[2]):
            raise IndexError('vertices outside the region volume map cube')

        # in memory data set
        if self.full_rmap_cache is not None:
            region_id = self.full_rmap_cache[x_plane, y_plane, z_plane]
            return region_id

        # not in memory have to go to disk
        slices = slice(x_plane, x_plane + 1), slice(y_plane, y_plane + 1), slice(z_plane, z_plane + 1)
        region_id = self.region_volume.read_data_slice(slices)[0, 0, 0]
        return region_id

    def _attempt_to_cache_regionmap(self, region_volume):
        a, b, c = region_volume.read_data_shape()
        if a * b * c <= 256 * 256 * 256:
            # read all
            slices = slice(a), slice(b), slice(c)
            self.full_rmap_cache = region_volume.read_data_slice(slices)
        else:
            self.full_rmap_cache = None

    def _base_before_launch(self, data_file, region_volume):
        if data_file is None:
            raise LaunchException("Please select a file to import")

        self.region_volume = region_volume

        if region_volume is not None:
            self._attempt_to_cache_regionmap(region_volume)
            # this is called here and not in _get_tract_region because it goes to disk to get this info
            # which would massively dominate the runtime of the import
            self.region_volume_shape = region_volume.read_data_shape()

        datatype = Tracts()
        datatype.storage_path = self.storage_path
        datatype.region_volume_map = region_volume
        return datatype


class _SpaceTransform(object):
    """
    Performs voxel to TVB space transformation
    """

    RAS_TO_TVB = numpy.array(
        [[0., 1., 0., 0.],
         [-1., 0., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]])

    def __init__(self, hdr):
        # this is an affine transform mapping the voxel space in which the tracts live to the surface space
        # see http://www.grahamwideman.com/gw/brain/fs/coords/fscoords.htm
        self.vox_to_ras = hdr['vox_to_ras']

        if self.vox_to_ras[3][3] == 0:
            # according to http://www.trackvis.org/docs/?subsect=fileformat this means that the matrix cannot be trusted
            self.vox_to_ras = numpy.eye(4)

    def transform(self, vertices):
        # to vox homogeneous coordinates
        w_coordinate = numpy.ones((1, len(vertices)), dtype=vertices.dtype)
        vertices = numpy.vstack([vertices.T, w_coordinate])
        # to RAS homogeneous space
        vertices = self.vox_to_ras.dot(vertices)
        # to TVB homogeneous space
        vertices = self.RAS_TO_TVB.dot(vertices)
        # to 3d space
        vertices = vertices.T[:, :3]
        return vertices


class TrackvizTractsImporter(_TrackImporterBase):
    """
    This imports tracts from the trackviz format
    """

    @transactional
    def launch(self, data_file, region_volume=None):
        datatype = self._base_before_launch(data_file, region_volume)

        # note the streaming parsing, we do not load the dataset in memory at once
        tract_gen, hdr = trackvis.read(data_file, as_generator=True)

        vox2ras = _SpaceTransform(hdr)
        tract_start_indices = [0]
        tract_region = []

        # we process tracts in bigger chunks to optimize disk write costs
        for tract_bundle in chunk_iter(tract_gen, self.READ_CHUNK):
            tract_bundle = [tr[0] for tr in tract_bundle]

            for tr in tract_bundle:
                tract_start_indices.append(tract_start_indices[-1] + len(tr))
                if region_volume is not None:
                    tract_region.append(self._get_tract_region(tr[0]))

            vertices = numpy.concatenate(tract_bundle)  # in voxel space
            vertices = vox2ras.transform(vertices)

            datatype.store_data_chunk("vertices", vertices, grow_dimension=0, close_file=False)

        datatype.tract_start_idx = tract_start_indices
        datatype.tract_region = numpy.array(tract_region, dtype=numpy.int16)
        return datatype


class ZipTxtTractsImporter(_TrackImporterBase):
    """
    This imports tracts from a zip containing txt files. One txt file for a tract.
    """
    _ui_name = "Tracts Zipped Txt"

    def get_form_class(self):
        return TrackZipImporterForm

    @transactional
    def launch(self, view_model):
        # type: (TrackImporterModel) -> [TractsIndex]
        region_volume_index = self.load_entity_by_gid(view_model.region_volume)
        datatype = self._base_before_launch(view_model.data_file, region_volume_index)

        tract_start_indices = [0]
        tract_region = []

        with TvbZip(view_model.data_file) as zipf:
            for tractf in sorted(zipf.namelist()):  # one track per file
                if not tractf.endswith('.txt'):  # omit directories and other non track files
                    continue
                vertices_file = zipf.open(tractf)
                tract_vertices = numpy.loadtxt(vertices_file, dtype=numpy.float32)

                tract_start_indices.append(tract_start_indices[-1] + len(tract_vertices))
                datatype.store_data_chunk("vertices", tract_vertices, grow_dimension=0, close_file=False)

                if region_volume_index is not None:
                    tract_region.append(self._get_tract_region(tract_vertices[0]))
                vertices_file.close()

        datatype.tract_start_idx = tract_start_indices
        datatype.tract_region = numpy.array(tract_region, dtype=numpy.int16)
        return datatype
