from tvb.core.neotraits.h5 import H5File, DataSet, Reference
from tvb.datatypes.tracts import Tracts


class TractsH5(H5File):
    def __init__(self, path):
        super(TractsH5, self).__init__(path)
        self.vertices = DataSet(Tracts.vertices, self, expand_dimension=0)
        self.tract_start_idx = DataSet(Tracts.tract_start_idx, self)
        self.tract_region = DataSet(Tracts.tract_region, self)
        self.region_volume_map = Reference(Tracts.region_volume_map, self)
