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
import colorsys
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.surfaces_framework import paths2url
from tvb.datatypes.tracts_data import TractData

LOG = get_logger(__name__)

TRACTS_CHUNK_SIZE = 100

class TractsFramework(TractData):
    __tablename__ = None

    def get_urls_for_rendering(self, sample_rate=1):
        sample_rate = int(sample_rate)
        url_vertices = []

        nchunks, rest = divmod(self.tracts_count, TRACTS_CHUNK_SIZE)
        if rest:
            nchunks +=1

        for i in xrange(nchunks):
            param_str = "slice_number=%d&sample_rate=%d" % (i, sample_rate)
            url4chunk = paths2url(self, 'get_tracts_slice', parameter=param_str , flatten=True)
            url_vertices.append(url4chunk)
        return url_vertices

    def _slice2range(self, slice_number):
        slice_number = int(slice_number)

        start = slice_number * TRACTS_CHUNK_SIZE
        stop = min(start + TRACTS_CHUNK_SIZE, self.tracts_count)
        return start, stop

    def get_tracts_slice(self, slice_number, sample_rate):
        """
        Returns a list of vertex arrays. One array per tract line.
        Sliced by tracts not vertices
        """
        sample_rate = int(sample_rate)
        # todo : range checks
        tracts = []
        start, stop = self._slice2range(slice_number)

        for i in xrange(start, stop):
            c = self.vertex_counts[i]
            # indexing warning
            tracts.append(list(self.vertices[i, :c:sample_rate * 3, :].flat))
        return tracts

    def get_tracts_color(self, slice_number, sample_rate):
        start, stop = self._slice2range(slice_number)
        hues = numpy.linspace(0, 1, self.tracts_count)[start:stop:sample_rate]
        return [colorsys.hls_to_rgb(h, 0.5, 0.5) for h in hues]