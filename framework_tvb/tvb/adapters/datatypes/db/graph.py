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
from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from tvb.datatypes.graph import Covariance, CorrelationCoefficients, ConnectivityMeasure
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataTypeMatrix
from tvb.core.neotraits.db import from_ndarray


class CovarianceIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=not Covariance.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    subtype = Column(String)

    def fill_from_has_traits(self, datatype):
        # type: (Covariance)  -> None
        super(CovarianceIndex, self).fill_from_has_traits(datatype)
        self.subtype = datatype.__class__.__name__
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.array_data)
        self.source_gid = datatype.source.gid.hex


class CorrelationCoefficientsIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid),
                        nullable=not CorrelationCoefficients.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    subtype = Column(String)

    labels_ordering = Column(String)

    def fill_from_has_traits(self, datatype):
        # type: (CorrelationCoefficients)  -> None
        super(CorrelationCoefficientsIndex, self).fill_from_has_traits(datatype)
        self.subtype = datatype.__class__.__name__
        self.labels_ordering = datatype.labels_ordering
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.array_data)
        self.source_gid = datatype.source.gid.hex


class ConnectivityMeasureIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    subtype = Column(String)

    connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid),
                              nullable=ConnectivityMeasure.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == connectivity_gid)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    def fill_from_has_traits(self, datatype):
        # type: (ConnectivityMeasure)  -> None
        super(ConnectivityMeasureIndex, self).fill_from_has_traits(datatype)
        self.subtype = datatype.__class__.__name__
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.array_data)
        self.connectivity_gid = datatype.connectivity.gid.hex
