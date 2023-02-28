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
import numpy
from tvb.core.neotraits.h5 import H5File, DataSet, Reference
from tvb.datatypes.tracts import Tracts


class TractsH5(H5File):
    MAX_N_VERTICES = 2 ** 16

    def __init__(self, path):
        super(TractsH5, self).__init__(path)
        self.vertices = DataSet(Tracts.vertices, self, expand_dimension=0)
        self.tract_start_idx = DataSet(Tracts.tract_start_idx, self)
        self.tract_region = DataSet(Tracts.tract_region, self)
        self.region_volume_map = Reference(Tracts.region_volume_map, self)

    def get_tract(self, i):
        """
        get a tract by index
        """
        start, end = self.tract_start_idx[i:i + 2]
        return self.vertices[start:end]

    def _get_tract_ids(self, region_id):
        tract_ids = numpy.where(self.tract_region.load() == region_id)[0]
        return tract_ids

    def _get_track_ids_webgl_chunks(self, region_id):
        """
        webgl can draw up to MAX_N_VERTICES vertices in a draw call.
        Assuming that no one track exceeds this limit we partition
        the tracts such that each track bundle has fewer than the max vertices
        :return: the id's of the tracts in a region chunked by the above criteria.
        """
        # We have to split the int64 range in many uint16 ranges
        tract_ids = self._get_tract_ids(region_id)

        tract_id_chunks = []
        chunk = []

        count = 0
        tidx = 0

        while tidx < len(tract_ids):  # tidx always grows
            tid = tract_ids[tidx]
            start, end = self.tract_start_idx[tid:tid + 2]
            track_len = end - start

            if track_len >= self.MAX_N_VERTICES:
                raise ValueError('Currently tracts are too long to be handled!')

            count += track_len

            if count < self.MAX_N_VERTICES:
                # add this track to the current chunk and advance to next track
                chunk.append(tid)
                tidx += 1
            else:
                # stay with the same track and start a new chunk
                tract_id_chunks.append(chunk)
                chunk = []
                count = 0

        if chunk:
            tract_id_chunks.append(chunk)

        return tract_id_chunks

    def get_vertices(self, region_id, slice_number=0):
        """
        Concatenates the vertices for all tracts starting in region_id.
        Returns a completely flat array as required by gl.bindBuffer apis
        """
        region_id = int(region_id)
        slice_number = int(slice_number)

        chunks = self._get_track_ids_webgl_chunks(region_id)
        tract_ids = chunks[slice_number]

        tracts_vertices = []
        for tid in tract_ids:
            tracts_vertices.append(self.get_tract(tid))

        self.close()

        if tracts_vertices:
            tracts_vertices = numpy.concatenate(tracts_vertices)
            return tracts_vertices.ravel()
        else:
            return numpy.array([])

    def get_line_starts(self, region_id):
        """
        Returns a compact representation of the element buffers required to draw the streams via gl.drawElements
        A list of indices that describe where the first vertex for a tract is in the vertex array returned by
        get_tract_vertices_starting_in_region
        """
        region_id = int(region_id)
        chunks = self._get_track_ids_webgl_chunks(region_id)
        chunk_line_starts = []
        tract_start_idx = self.tract_start_idx  # traits make the . expensive

        for tract_ids in chunks:
            offset = 0
            tract_offsets = [0]

            for tid in tract_ids:
                start, end = tract_start_idx[tid:tid + 2]
                track_len = end - start
                offset += track_len
                tract_offsets.append(offset)

            chunk_line_starts.append(tract_offsets)

        return chunk_line_starts

    def write_vertices_slice(self, partial_result):
        """
        Append a new value to the ``vertices`` attribute.
        """
        self.vertices.append(partial_result, close_file=False)
