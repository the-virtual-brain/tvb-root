from tvb.core.neotraits.h5 import H5File, DataSet, Reference, Json
from tvb.datatypes.temporal_correlations import CrossCorrelation


class CrossCorrelationH5(H5File):

    def __init__(self, path):
        super(CrossCorrelationH5, self).__init__(path)
        self.array_data = DataSet(CrossCorrelation.array_data, self, expand_dimension=3)
        self.source = Reference(CrossCorrelation.source, self)
        self.time = DataSet(CrossCorrelation.time, self)
        self.labels_ordering = Json(CrossCorrelation.labels_ordering, self)
