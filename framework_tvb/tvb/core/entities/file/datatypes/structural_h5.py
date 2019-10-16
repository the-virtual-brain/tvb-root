from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference
from tvb.datatypes.structural import StructuralMRI


class StructuralMRIH5(H5File):

    def __init__(self, path):
        super(StructuralMRIH5, self).__init__(path)
        self.array_data = DataSet(StructuralMRI.array_data, self)
        self.weighting = Scalar(StructuralMRI.weighting, self)
        self.volume = Reference(StructuralMRI.volume, self)