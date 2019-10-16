from tvb.core.neotraits.h5 import H5File, DataSet, Reference
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping


class RegionMappingH5(H5File):

    def __init__(self, path):
        super(RegionMappingH5, self).__init__(path)
        self.array_data = DataSet(RegionMapping.array_data, self)
        self.connectivity = Reference(RegionMapping.connectivity, self)
        self.surface = Reference(RegionMapping.surface, self)


class RegionVolumeMappingH5(H5File):

    def __init__(self, path):
        super(RegionVolumeMappingH5, self).__init__(path)
        self.array_data = DataSet(RegionVolumeMapping.array_data, self)
        self.connectivity = Reference(RegionVolumeMapping.connectivity, self)
        self.volume = Reference(RegionVolumeMapping.volume, self)
