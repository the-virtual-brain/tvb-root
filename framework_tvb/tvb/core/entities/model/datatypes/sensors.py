from sqlalchemy import Column, Integer, ForeignKey, String

from tvb.core.neotraits.db import HasTraitsIndex


class SensorsIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)
    number_of_sensors = Column(Integer, nullable=False)
    sensors_type = Column(String, nullable=False)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.number_of_sensors = datatype.number_of_sensors
        self.sensors_type = datatype.sensors_type
