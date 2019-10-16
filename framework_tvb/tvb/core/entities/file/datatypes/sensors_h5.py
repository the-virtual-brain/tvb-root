from tvb.core.neotraits.h5 import H5File, DataSet, Scalar
from tvb.datatypes.sensors import Sensors


class SensorsH5(H5File):
    def __init__(self, path):
        super(SensorsH5, self).__init__(path)
        self.sensors_type = Scalar(Sensors.sensors_type, self)
        self.labels = DataSet(Sensors.labels, self)
        self.locations = DataSet(Sensors.locations, self)
        self.has_orientation = Scalar(Sensors.has_orientation, self)
        self.orientations = DataSet(Sensors.orientations, self)
        self.number_of_sensors = Scalar(Sensors.number_of_sensors, self)
        self.usable = DataSet(Sensors.usable, self)

    def get_locations(self):
        return self.locations.load()

    def get_labels(self):
        return self.labels.load()