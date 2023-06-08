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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
from sqlalchemy import Column, Integer, ForeignKey, String, Boolean
from sqlalchemy.orm import relationship
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex, RegionMappingIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataTypeMatrix
from tvb.core.entities.storage import dao
from tvb.datatypes.graph import Covariance, CorrelationCoefficients, ConnectivityMeasure


class CovarianceIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    fk_source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=not Covariance.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=fk_source_gid, primaryjoin=TimeSeriesIndex.gid == fk_source_gid)

    def fill_from_has_traits(self, datatype):
        # type: (Covariance)  -> None
        super(CovarianceIndex, self).fill_from_has_traits(datatype)
        self.fk_source_gid = datatype.source.gid.hex


class CorrelationCoefficientsIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    fk_source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid),
                           nullable=not CorrelationCoefficients.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=fk_source_gid, primaryjoin=TimeSeriesIndex.gid == fk_source_gid)

    labels_ordering = Column(String)

    def get_extra_info(self):
        labels_dict = {}
        labels_dict["labels_ordering"] = self.source.labels_ordering
        labels_dict["labels_dimensions"] = self.source.labels_dimensions
        return labels_dict

    def fill_from_has_traits(self, datatype):
        # type: (CorrelationCoefficients)  -> None
        super(CorrelationCoefficientsIndex, self).fill_from_has_traits(datatype)
        self.labels_ordering = json.dumps(datatype.labels_ordering)
        self.fk_source_gid = datatype.source.gid.hex


class ConnectivityMeasureIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    fk_connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid),
                                 nullable=ConnectivityMeasure.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=fk_connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == fk_connectivity_gid)

    has_surface_mapping = Column(Boolean, nullable=False, default=False)

    def fill_from_h5(self, h5_file):
        super(ConnectivityMeasureIndex, self).fill_from_h5(h5_file)
        self.fk_connectivity_gid = h5_file.connectivity.load().hex
        self.title = h5_file.title.load()
        self.has_volume_mapping = False
        self.has_surface_mapping = False
        rm_list = dao.get_generic_entity(RegionMappingIndex, self.fk_connectivity_gid, 'fk_connectivity_gid')
        if rm_list:
            self.has_surface_mapping = True

        rvm_list = dao.get_generic_entity(RegionVolumeMappingIndex, self.fk_connectivity_gid, 'fk_connectivity_gid')
        if rvm_list:
            self.has_volume_mapping = True

    def fill_from_has_traits(self, datatype):
        # type: (ConnectivityMeasure)  -> None
        super(ConnectivityMeasureIndex, self).fill_from_has_traits(datatype)
        self.fk_connectivity_gid = datatype.connectivity.gid.hex
        self.title = datatype.title

        self.has_volume_mapping = False
        self.has_surface_mapping = False

        no_reg = datatype.connectivity.number_of_regions
        if not (no_reg in self.parsed_shape):
            return

        rm_list = dao.get_generic_entity(RegionMappingIndex, self.fk_connectivity_gid, 'fk_connectivity_gid')
        if rm_list:
            self.has_surface_mapping = True

        rvm_list = dao.get_generic_entity(RegionVolumeMappingIndex, self.fk_connectivity_gid, 'fk_connectivity_gid')
        if rvm_list:
            self.has_volume_mapping = True


    @property
    def display_name(self):
        """
        Overwrite from superclass and add number of regions field
        """
        result = super(ConnectivityMeasureIndex, self).display_name
        if self.title:
            result = result + " - " + (self.title if len(self.title) < 50 else self.title[:46] + "...")
        return result
