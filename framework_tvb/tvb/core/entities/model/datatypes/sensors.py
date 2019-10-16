from sqlalchemy import Column, Integer, ForeignKey, String

from tvb.core.entities.model.model_datatype import DataType


class SensorsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
    number_of_sensors = Column(Integer, nullable=False)
    sensors_type = Column(String, nullable=False)

    def fill_from_has_traits(self, datatype):
        self.number_of_sensors = datatype.number_of_sensors
        self.sensors_type = datatype.sensors_type
