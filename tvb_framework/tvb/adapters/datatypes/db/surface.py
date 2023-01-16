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

from tvb.datatypes.surfaces import Surface
from sqlalchemy import Column, Integer, ForeignKey, Float, Boolean, String
from tvb.core.entities.model.model_datatype import DataType


class SurfaceIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    surface_type = Column(String, nullable=False)
    valid_for_simulations = Column(Boolean, nullable=False)
    number_of_vertices = Column(Integer, nullable=False)
    number_of_triangles = Column(Integer, nullable=False)
    number_of_edges = Column(Integer, nullable=False)
    bi_hemispheric = Column(Boolean, nullable=False)
    edge_mean_length = Column(Float, nullable=False)
    edge_min_length = Column(Float, nullable=False)
    edge_max_length = Column(Float, nullable=False)

    def fill_from_has_traits(self, datatype):
        # type: (Surface)  -> None
        super(SurfaceIndex, self).fill_from_has_traits(datatype)
        self.surface_type = datatype.surface_type
        self.valid_for_simulations = datatype.valid_for_simulations
        self.number_of_vertices = datatype.number_of_vertices
        self.number_of_triangles = datatype.number_of_triangles
        self.number_of_edges = datatype.number_of_edges
        self.bi_hemispheric = datatype.bi_hemispheric
        self.edge_mean_length = datatype.edge_mean_length
        self.edge_min_length = datatype.edge_min_length
        self.edge_max_length = datatype.edge_max_length

    def get_subtype_attr(self):
        return self.surface_type

    @property
    def display_name(self):
        """
        Overwrite from superclass and subtype
        """
        previous = "Surface"
        return previous + " - " + str(self.surface_type)
