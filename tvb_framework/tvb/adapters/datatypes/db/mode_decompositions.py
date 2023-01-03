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
import numpy
from sqlalchemy import Column, Integer, ForeignKey, String, Boolean
from sqlalchemy.orm import relationship
from tvb.datatypes.mode_decompositions import PrincipalComponents, IndependentComponents
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataType


class PrincipalComponentsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    fk_source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid),
                           nullable=not PrincipalComponents.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=fk_source_gid, primaryjoin=TimeSeriesIndex.gid == fk_source_gid)

    def fill_from_has_traits(self, datatype):
        # type: (PrincipalComponents)  -> None
        super(PrincipalComponentsIndex, self).fill_from_has_traits(datatype)
        self.fk_source_gid = datatype.source.gid.hex

    def get_extra_info(self):
        labels_dict = {}
        labels_dict["labels_ordering"] = self.source.labels_ordering
        labels_dict["labels_dimensions"] = self.source.labels_dimensions
        return labels_dict


class IndependentComponentsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    fk_source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid),
                           nullable=not PrincipalComponents.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=fk_source_gid, primaryjoin=TimeSeriesIndex.gid == fk_source_gid)

    ndim = Column(Integer, default=0)
    shape = Column(String, nullable=True)
    array_has_complex = Column(Boolean, default=False)

    def fill_from_has_traits(self, datatype):
        # type: (IndependentComponents)  -> None
        super(IndependentComponentsIndex, self).fill_from_has_traits(datatype)
        self.fk_source_gid = datatype.source.gid.hex

        self.shape = json.dumps(datatype.unmixing_matrix.shape)
        self.ndim = len(datatype.unmixing_matrix.shape)
        self.array_has_complex = numpy.iscomplex(datatype.unmixing_matrix).any().item()

    def get_extra_info(self):
        labels_dict = {}
        labels_dict["labels_ordering"] = self.source.labels_ordering
        labels_dict["labels_dimensions"] = self.source.labels_dimensions
        return labels_dict

    @property
    def parsed_shape(self):
        try:
            return tuple(json.loads(self.shape))
        except:
            return ()
