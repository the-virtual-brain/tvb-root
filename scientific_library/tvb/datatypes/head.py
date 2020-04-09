# coding=utf-8

from collections import OrderedDict

from six import string_types
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.sensors import Sensors
from tvb.datatypes.surfaces import CorticalSurface
from tvb.simulator.plot.utils.data_structures_utils import isequal_string, is_integer
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error, warning


class Head:
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """
    logger = initialize_logger(__name__)
    folderpath = ""

    def __init__(self, connectivity, sensors=OrderedDict(),
                 cortical_surface=None, subcortical_surface=None,
                 cortical_region_mapping=None, subcortical_region_mapping=None,
                 region_volume_mapping=None, t1=None, name='', folderpath=""):
        self.name = name
        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.subcortical_surface = subcortical_surface
        self.cortical_region_mapping = cortical_region_mapping
        self.subcortical_region_mapping = subcortical_region_mapping
        self.region_volume_mapping = region_volume_mapping
        self.t1 = t1
        self.sensors = sensors
        self.folderpath = folderpath

    @property
    def number_of_regions(self):
        return self.connectivity.number_of_regions

    @property
    def cortex(self):
        cortex = Cortex()
        cortex.region_mapping_data = self.cortical_region_mapping
        cortex = cortex.populate_cortex(self.cortical_surface._tvb, {})
        for s_type, sensors in self.sensors.items():
            if isinstance(sensors, OrderedDict) and len(sensors) > 0:
                projection = sensors.values()[0]
                if projection is not None:
                    setattr(cortex, s_type.lower(), projection.projection_data)
        cortex.configure()
        return cortex

    def configure(self):
        if isinstance(self.connectivity, Connectivity):
            self.connectivity.configure()
        if isinstance(self.cortical_surface, CorticalSurface):
            self.cortical_surface.configure()
        if isinstance(self.cortical_surface, CorticalSurface):
            self.subcortical_surface.configure()
        for s_type, sensors_set in self.sensors.items():
            for sensor, projection in sensors_set.items():
                if isinstance(sensor, Sensors):
                    sensor.configure()

    def filter_regions(self, filter_arr):
        return self.connectivity.region_labels[filter_arr]

    def get_sensors(self, s_type=SensorTypes.TYPE_EEG.value, name_or_index=None):
        sensors_set = OrderedDict()
        if s_type not in SensorTypesNames:
            raise_value_error("Invalid input sensor type!: %s" % str(s_type))
        else:
            sensors_set = self.sensors.get(s_type, None)
        out_sensor = None
        out_projection = None
        if isinstance(sensors_set, OrderedDict):
            if isinstance(name_or_index, string_types):
                for sensor, projection in sensors_set.items():
                    if isequal_string(sensor.name, name_or_index):
                        out_sensor = sensor
                        out_projection = projection
            elif is_integer(name_or_index):
                out_sensor = sensors_set.keys()[name_or_index]
                out_projection = sensors_set.values()[name_or_index]
            else:
                return sensors_set
        return out_sensor, out_projection

    def set_sensors(self, input_sensors, s_type=SensorTypes.TYPE_EEG.value, reset=False):
        if not isinstance(input_sensors, (Sensors, dict, list, tuple)):
            return raise_value_error("Invalid input sensors instance''!: %s" % str(input_sensors))
        if s_type not in SensorTypesNames:
            raise_value_error("Invalid input sensor type!: %s" % str(s_type))
        sensors_set = self.get_sensors(s_type)[0]
        if reset is True:
            sensors_set = OrderedDict()
        if isinstance(input_sensors, dict):
            input_projections = input_sensors.values()
            input_sensors = input_sensors.keys()
        else:
            if isinstance(input_sensors, Sensors):
                input_sensors = [input_sensors]
            else:
                input_sensors = list(input_sensors)
            input_projections = [None] * len(input_sensors)

        for sensor, projection in zip(input_sensors, input_projections):
            if not isinstance(sensor, Sensors):
                raise_value_error("Input sensors:\n%s"
                                  "\nis not a valid Sensors object of type %s!" % (str(sensor), s_type))
            if sensor.sensors_type != s_type:
                raise_value_error("Sensors %s \nare not of type %s!" % (str(sensor), s_type))
            if not isinstance(projection, ProjectionMatrix):
                warning("projection is not set for sensor with name:\n%s!" % sensor.name)
                sensors_set.update({sensor: None})
            else:
                if projection.projection_type != SensorTypesToProjectionDict[s_type]:
                    raise_value_error("Disaggreement between sensors'type %s and projection's type %s!"
                                      % (sensor.sensors_type, projection.projection_type))
                good_sensor_shape = (sensor.number_of_sensors, self.number_of_regions)
                if projection.projection_data.shape != good_sensor_shape:
                    warning("projections' shape %s of sensor %s "
                            "is not equal to (number_of_sensors, number_of_regions)=%s!"
                            % (str(projection.projection_data.shape), sensor.name, str(good_sensor_shape)))
                sensors_set.update({sensor: projection})

        self.sensors[s_type] = sensors_set
