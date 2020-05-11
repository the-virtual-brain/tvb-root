from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.patterns import StimuliSurfaceIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class ConnectivityRemover(ABCRemover):
    """
    Connectivity specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Connectivity is to be removed.
        """
        if not skip_validation:
            associated_ts = dao.get_generic_entity(TimeSeriesRegionIndex, self.handled_datatype.gid, 'fk_connectivity_gid')
            associated_rm = dao.get_generic_entity(RegionMappingIndex, self.handled_datatype.gid, 'fk_connectivity_gid')
            associated_stim = dao.get_generic_entity(StimuliSurfaceIndex, self.handled_datatype.gid, 'fk_connectivity_gid')
            associated_mes = dao.get_generic_entity(ConnectivityMeasureIndex, self.handled_datatype.gid, 'fk_connectivity_gid')
            msg = "Connectivity cannot be removed as it is used by at least one "

            if len(associated_ts) > 0:
                raise RemoveDataTypeException(msg + " TimeSeriesRegion.")
            if len(associated_rm) > 0:
                raise RemoveDataTypeException(msg + " RegionMapping.")
            if len(associated_stim) > 0:
                raise RemoveDataTypeException(msg + " StimuliRegion.")
            if len(associated_mes) > 0:
                raise RemoveDataTypeException(msg + " ConnectivityMeasure.")

        # Update child Connectivities, if any. ?

        ABCRemover.remove_datatype(self, skip_validation)