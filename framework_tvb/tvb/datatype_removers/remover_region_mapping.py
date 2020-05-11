from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException



class RegionMappingRemover(ABCRemover):
    """
    RegionMapping specific validations at remove time.
    """
    FIELD_NAME = "fk_region_mapping_gid"
    CLASS_NAME = "RegionMappingIndex"

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Sensor is to be removed.
        """
        if not skip_validation:
            tsr = dao.get_generic_entity(TimeSeriesRegionIndex, self.handled_datatype.gid, self.FIELD_NAME)
            error_msg = "%s cannot be removed because is still used by %d TimeSeries Region entities."
            if tsr:
                raise RemoveDataTypeException(error_msg % (self.CLASS_NAME, len(tsr)))

        ABCRemover.remove_datatype(self, skip_validation)



class RegionVolumeMappingRemover(RegionMappingRemover):
    """
    RegionVolumeMapping specific validations at remove time.
    """

    FIELD_NAME = "fk_region_mapping_volume_gid"
    CLASS_NAME = "RegionVolumeMappingIndex"