from tvb.core.neotraits.h5 import H5File, DataSet, Reference, Json
from tvb.datatypes.temporal_correlations import CrossCorrelation


class CrossCorrelationH5(H5File):

    def __init__(self, path):
        super(CrossCorrelationH5, self).__init__(path)
        self.array_data = DataSet(CrossCorrelation.array_data, expand_dimension=3)
        self.source = Reference(CrossCorrelation.source)
        self.time = DataSet(CrossCorrelation.time)
        self.labels_ordering = Json(CrossCorrelation.labels_ordering)
        self._end_accessor_declarations()


    def read_data_shape(self):
        """
        The shape of the data
        """
        return self.array_data.shape

    def read_data_slice(self, data_slice):
        """
        Expose chunked-data access.
        """
        return self.array_data[data_slice]

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.array_data.append(partial_result.array_data)

