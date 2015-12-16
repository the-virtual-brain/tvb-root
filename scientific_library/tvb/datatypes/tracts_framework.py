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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
module docstring
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.tracts_data import TractData

LOG = get_logger(__name__)

TRACTS_CHUNK_SIZE = 100

class TractsFramework(TractData):
    __tablename__ = None

    def get_tract(self, i):
        """
        get a tract by index
        """
        start, end = self.tract_start_idx[i:i+2]
        return self.get_data('vertices', slice(start, end), close_file=False)


    def _get_tract_ids(self, region_id):
        tract_ids = numpy.where(self.tract_region == region_id)[0]
        return tract_ids


    def get_tracts_starting_in_region(self, region_id):
        region_id = int(region_id)
        tract_ids = self._get_tract_ids(region_id)

        tracts = []
        for tid in tract_ids:
            tracts.append(self.get_tract(tid).tolist())

        self.close_file()

        return tracts


    def get_vertices(self, region_id):
        """
        Concatenates the vertices for all tracts starting in region_id.
        Returns a completely flat array as required by gl.bindBuffer apis
        """
        region_id = int(region_id)
        tract_ids = self._get_tract_ids(region_id)

        tracts_vertices = []
        for tid in tract_ids:
            tracts_vertices.append(self.get_tract(tid))

        self.close_file()
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
        tract_ids = self._get_tract_ids(region_id)

        offset = 0
        tract_offsets = [0]

        for tid in tract_ids:
            start, end = self.tract_start_idx[tid:tid+2]
            track_len = end - start
            offset +=track_len
            tract_offsets.append(offset)

        return tract_offsets
