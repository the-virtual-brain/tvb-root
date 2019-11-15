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
from tvb.basic.neotraits.api import Attr
from tvb.datatypes.patterns import StimuliRegion, StimuliSurface
from tvb.core.neotraits.h5 import H5File, Reference, DataSet, Scalar


class StimuliRegionH5(H5File):

    def __init__(self, path):
        super(StimuliRegionH5, self).__init__(path)
        self.spatial = Reference(StimuliRegion.spatial, self)
        self.temporal = Reference(StimuliRegion.temporal, self)
        self.connectivity = Reference(StimuliRegion.connectivity, self)
        self.weight = DataSet(StimuliRegion.weight, self)

    def store(self, datatype, scalars_only=False):
        super(StimuliRegionH5, self).store(datatype, scalars_only)
        self.connectivity.store(datatype.connectivity)
        self.spatial.store(datatype.spatial.gid)
        self.temporal.store(datatype.temporal.gid)


class StimuliSurfaceH5(H5File):

    def __init__(self, path):
        super(StimuliSurfaceH5, self).__init__(path)
        self.spatial = Scalar(Attr(str), self, name='spatial')
        self.temporal = Scalar(Attr(str), self, name='temporal')
        self.surface = Reference(StimuliSurface.surface, self)
        self.focal_points_surface = DataSet(StimuliSurface.focal_points_surface, self)
        self.focal_points_triangles = DataSet(StimuliSurface.focal_points_triangles, self)

    def store(self, datatype, scalars_only=False):
        self.surface.store(datatype.surface)
        self.focal_points_surface.store(datatype.focal_points_surface)
        self.focal_points_triangles.store(datatype.focal_points_triangles)
        self.spatial.store(datatype.spatial.to_json(datatype.spatial))
        self.temporal.store(datatype.temporal.to_json(datatype.temporal))

    def load_into(self, datatype):
        datatype.gid = self.gid.load()
        datatype.surface = self.surface.load()
        datatype.focal_points_triangles = self.focal_points_triangles.load()
        datatype.focal_points_surface = self.focal_points_surface.load()
        spatial_eq = self.spatial.load()
        spatial_eq = datatype.spatial.from_json(spatial_eq)
        datatype.spatial = spatial_eq
        temporal_eq = self.temporal.load()
        temporal_eq = datatype.temporal.from_json(temporal_eq)
        datatype.temporal = temporal_eq
