from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class SensorRemover(ABCRemover):
    """
    Sensor specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Sensor is to be removed.
        """
        if not skip_validation:
            projection_matrices = dao.get_generic_entity(ProjectionMatrixIndex, self.handled_datatype.gid, 'fk_sensors_gid')
            error_msg = "Sensor cannot be removed because is still used by %d Projection Matrix entities."
            if projection_matrices:
                raise RemoveDataTypeException(error_msg % len(projection_matrices))

        ABCRemover.remove_datatype(self, skip_validation)