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

    def get_grouped_space_labels(self):
        """
        :return: A list [('left', [lh_labels)], ('right': [rh_labels])]
        """
        hemispheres = self.hemispheres.load()
        region_labels = self.region_labels.load()
        if hemispheres is not None and hemispheres.size:
            l, r = [], []

            for i, (is_right, label) in enumerate(zip(hemispheres, region_labels)):
                if is_right:
                    r.append((i, label))
                else:
                    l.append((i, label))
            return [('left', l), ('right', r)]
        else:
            return [('', list(enumerate(region_labels)))]

    def get_default_selection(self):
        # should this be sub-selection or all always?
        sel = self.saved_selection.load()
        if sel is not None and len(sel) > 0:
            return sel
        else:
            return range(len(self.region_labels.load()))

    def get_measure_points_selection_gid(self):
        """
        :return: the associated connectivity gid
        """
        return self.gid.load().hex