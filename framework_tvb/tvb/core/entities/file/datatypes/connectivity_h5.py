from tvb.core.neotraits._h5accessors import Json
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar
from tvb.datatypes.connectivity import Connectivity


class ConnectivityH5(H5File):
    def __init__(self, path):
        super(ConnectivityH5, self).__init__(path)
        self.region_labels = DataSet(Connectivity.region_labels, self)
        self.weights = DataSet(Connectivity.weights, self)
        self.undirected = Scalar(Connectivity.undirected, self)
        self.tract_lengths = DataSet(Connectivity.tract_lengths, self)
        self.centres = DataSet(Connectivity.centres, self)
        self.cortical = DataSet(Connectivity.cortical, self)
        self.hemispheres = DataSet(Connectivity.hemispheres, self)
        self.orientations = DataSet(Connectivity.orientations, self)
        self.areas = DataSet(Connectivity.areas, self)
        self.number_of_regions = Scalar(Connectivity.number_of_regions, self)
        self.number_of_connections = Scalar(Connectivity.number_of_connections, self)
        self.parent_connectivity = Scalar(Connectivity.parent_connectivity, self)
        self.saved_selection = Json(Connectivity.saved_selection, self)