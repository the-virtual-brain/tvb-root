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
from sqlalchemy import Column, Integer, ForeignKey, Boolean, Float
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.neotraits.db import from_ndarray
from tvb.datatypes.connectivity import Connectivity


class ConnectivityIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    number_of_regions = Column(Integer, nullable=False)
    number_of_connections = Column(Integer, nullable=False)
    undirected = Column(Boolean)

    weights_min = Column(Float)
    weights_max = Column(Float)
    weights_mean = Column(Float)

    tract_lengths_min = Column(Float)
    tract_lengths_max = Column(Float)
    tract_lengths_mean = Column(Float)

    has_cortical_mask = Column(Boolean)
    has_hemispheres_mask = Column(Boolean)

    def fill_from_has_traits(self, datatype):
        # type: (Connectivity)  -> None
        super(ConnectivityIndex, self).fill_from_has_traits(datatype)

        self.has_cortical_mask = datatype.cortical is not None
        self.has_hemispheres_mask = datatype.hemispheres is not None
        self.number_of_regions = datatype.number_of_regions
        self.number_of_connections = datatype.number_of_connections
        self.undirected = datatype.undirected
        self.weights_min, self.weights_max, self.weights_mean = from_ndarray(datatype.weights)
        self.tract_lengths_min, self.tract_lengths_max, self.tract_lengths_mean = from_ndarray(datatype.tract_lengths)

    @property
    def display_name(self):
        """
        Overwrite from superclass and add number of regions field
        """
        previous = "Connectivity"
        return previous + " [" + str(self.number_of_regions) + "]"

    @staticmethod
    def accepted_filters():
        filters = DataType.accepted_filters()
        filters.update({FilterChain.datatype + '.number_of_regions':
                            {'type': 'int', 'display': 'No of Regions', 'operations': ['==', '<', '>']}})
        return filters
