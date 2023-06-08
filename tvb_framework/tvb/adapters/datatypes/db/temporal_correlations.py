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

import json
from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataTypeMatrix
from tvb.datatypes.temporal_correlations import CrossCorrelation


class CrossCorrelationIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    fk_source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=not CrossCorrelation.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=fk_source_gid, primaryjoin=TimeSeriesIndex.gid == fk_source_gid)

    labels_ordering = Column(String, nullable=False)

    def get_extra_info(self):
        labels_dict = {}
        labels_dict["labels_ordering"] = self.source.labels_ordering
        labels_dict["labels_dimensions"] = self.source.labels_dimensions
        return labels_dict

    def fill_from_has_traits(self, datatype):
        # type: (CrossCorrelation)  -> None
        super(CrossCorrelationIndex, self).fill_from_has_traits(datatype)
        self.labels_ordering = json.dumps(datatype.labels_ordering)
        self.fk_source_gid = datatype.source.gid.hex
