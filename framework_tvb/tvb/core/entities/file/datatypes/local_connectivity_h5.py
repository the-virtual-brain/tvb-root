from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference
from tvb.datatypes.local_connectivity import LocalConnectivity


class LocalConnectivityH5(H5File):
    def __init__(self, path):
        super(LocalConnectivityH5, self).__init__(path)
        self.surface = Reference(LocalConnectivity.surface, self)
        self.matrix = DataSet(LocalConnectivity.matrix, self)  # This is Attr in Datatype, dataset in H5
        self.equation = Scalar(LocalConnectivity.equation, self)
        self.cutoff = Scalar(LocalConnectivity.cutoff, self)
        # H5 has more datasets than matrix
