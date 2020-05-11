from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.datatypes.db.patterns import StimuliSurfaceIndex
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
            error_msg = "Surface cannot be removed because is still used by a "

            if len(associated_ts) > 0:
                raise RemoveDataTypeException(error_msg + " TimeSeriesSurface.")
            if len(associated_rm) > 0:
                raise RemoveDataTypeException(error_msg + " RegionMapping.")
            if len(associated_lc) > 0:
                raise RemoveDataTypeException(error_msg + " LocalConnectivity.")
            if len(associated_stim) > 0:
                raise RemoveDataTypeException(error_msg + " StimuliSurfaceData.")

        ABCRemover.remove_datatype(self, skip_validation)