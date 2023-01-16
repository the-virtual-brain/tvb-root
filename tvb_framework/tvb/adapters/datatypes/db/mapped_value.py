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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json

from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.adapters.datatypes.h5.mapped_value_h5 import ValueWrapper
from tvb.core.entities.file.simulator.datatype_measure_h5 import DatatypeMeasure
from tvb.core.entities.model.model_datatype import DataType


class ValueWrapperIndex(DataType):
    """
    Class to wrap a singular value storage in DB.
    """
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
    data_value = Column(String)
    data_type = Column(String, default='unknown')
    data_name = Column(String)

    @property
    def display_name(self):
        """ Simple String to be used for display in UI."""
        return "Value Wrapper - " + self.data_name + " : " + str(self.data_value) + " (" + str(self.data_type) + ")"

    def fill_from_has_traits(self, datatype):
        # type: (ValueWrapper)  -> None
        super(ValueWrapperIndex, self).fill_from_has_traits(datatype)
        self.data_value = datatype.data_value
        self.data_type = datatype.data_type
        self.data_name = datatype.data_name


class DatatypeMeasureIndex(DataType):
    """
    Class to hold the metric for a previous stored DataType.
    E.g. Measure (single value) for any TimeSeries resulted in a group of Simulations
    """
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
    # Actual measure (dictionary Algorithm: single Value) serialized
    metrics = Column(String)
    # DataType for which the measure was computed.
    fk_source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=False)
    source = relationship(TimeSeriesIndex, foreign_keys=fk_source_gid, primaryjoin=TimeSeriesIndex.gid == fk_source_gid)

    @property
    def display_name(self):

        result = super(DatatypeMeasureIndex, self).display_name

        if self.metrics is not None:
            metrics = json.loads(self.metrics)
            for entry, metric_value in metrics.items():
                result = result + " -- " + entry + ' : ' + str(metric_value)

        return result

    def fill_from_has_traits(self, datatype):
        # type: (DatatypeMeasure)  -> None
        super(DatatypeMeasureIndex, self).fill_from_has_traits(datatype)
        self.metrics = json.dumps(datatype.metrics)
        self.fk_source_gid = datatype.analyzed_datatype.gid.hex
