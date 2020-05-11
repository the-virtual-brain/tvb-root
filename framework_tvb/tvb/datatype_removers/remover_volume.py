from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException

class VolumeRemover(ABCRemover):
    """
    Surface specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Surface is to be removed.
        """
        if not skip_validation:
            associated_ts = dao.get_generic_entity(TimeSeriesVolumeIndex, self.handled_datatype.gid, 'fk_volume_gid')
            # do we still need Spatial?...
            # associated_stim = dao.get_generic_entity(SpatialPatternVolume, self.handled_datatype.gid, 'fk_volume_gid')
            error_msg = "Surface cannot be removed because is still used by a "
            if len(associated_ts) > 0:
                raise RemoveDataTypeException(error_msg + " TimeSeriesVolumeIndex.")
            # if len(associated_stim) > 0:
            #     raise RemoveDataTypeException(error_msg + " SpatialPatternVolumeIndex.")

        ABCRemover.remove_datatype(self, skip_validation)