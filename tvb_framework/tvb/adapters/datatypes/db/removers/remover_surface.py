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

from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.patterns import StimuliSurfaceIndex
from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesSurfaceIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class SurfaceRemover(ABCRemover):
    """
    Surface specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Surface is to be removed.
        """
        if not skip_validation:
            associated_ts = dao.get_generic_entity(TimeSeriesSurfaceIndex, self.handled_datatype.gid, 'fk_surface_gid')
            associated_rm = dao.get_generic_entity(RegionMappingIndex, self.handled_datatype.gid, 'fk_surface_gid')
            associated_lc = dao.get_generic_entity(LocalConnectivityIndex, self.handled_datatype.gid, 'fk_surface_gid')
            associated_stim = dao.get_generic_entity(StimuliSurfaceIndex, self.handled_datatype.gid, 'fk_surface_gid')
            associated_pms = dao.get_generic_entity(ProjectionMatrixIndex, self.handled_datatype.gid,
                                                    'fk_brain_skull_gid')
            associated_pms.extend(dao.get_generic_entity(ProjectionMatrixIndex, self.handled_datatype.gid,
                                                         'fk_skull_skin_gid'))
            associated_pms.extend(dao.get_generic_entity(ProjectionMatrixIndex, self.handled_datatype.gid,
                                                         'fk_skin_air_gid'))
            associated_pms.extend(dao.get_generic_entity(ProjectionMatrixIndex, self.handled_datatype.gid,
                                                         'fk_source_gid'))
            error_msg = "Surface cannot be removed because is still used by %d %s entities "

            if len(associated_ts) > 0:
                raise RemoveDataTypeException(error_msg % (len(associated_ts), "TimeSeriesSurface"))
            if len(associated_rm) > 0:
                raise RemoveDataTypeException(error_msg % (len(associated_rm), "RegionMapping"))
            if len(associated_lc) > 0:
                raise RemoveDataTypeException(error_msg % (len(associated_lc), " LocalConnectivity"))
            if len(associated_stim) > 0:
                raise RemoveDataTypeException(error_msg % (len(associated_stim), "StimuliSurface"))
            if len(associated_pms) > 0:
                raise RemoveDataTypeException(error_msg % (len(associated_pms), "ProjectionMatrix"))

        ABCRemover.remove_datatype(self, skip_validation)
