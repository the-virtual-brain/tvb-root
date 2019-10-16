from tvb.core.neotraits.h5 import H5File, DataSet, Scalar
from tvb.datatypes.volumes import Volume


class VolumeH5(H5File):

    def __init__(self, path):
        super(VolumeH5, self).__init__(path)
        self.origin = DataSet(Volume.origin, self)
        self.voxel_size = DataSet(Volume.voxel_size, self)
        self.voxel_unit = Scalar(Volume.voxel_unit, self)
