# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
from sqlalchemy.orm import relationship
from sqlalchemy import Column, String, Integer, ForeignKey
from tvb.core.entities.model.model_datatype import DataType
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex


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


class DatatypeMeasureIndex(DataType):
    """
    Class to hold the metric for a previous stored DataType.
    E.g. Measure (single value) for any TimeSeries resulted in a group of Simulations
    """
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
    # Actual measure (dictionary Algorithm: single Value) serialized
    metrics = Column(String)
    # DataType for which the measure was computed.
    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=False)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    @property
    def display_name(self):
        """
        To be implemented in each sub-class which is about to be displayed in UI,
        and return the text to appear.
        """
        name = "-"
        if self.metrics is not None:
            value = "\n"
            metrics = json.loads(self.metrics)
            for entry, metric_value in metrics.items():
                value = value + entry + ' : ' + str(metric_value) + '\n'
            name = value
        return name
