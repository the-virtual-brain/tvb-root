from tvb.datatypes.graph import Covariance, CorrelationCoefficients, ConnectivityMeasure

from tvb.core.neotraits.h5 import H5File, DataSet, Reference, Json


class CovarianceH5(H5File):

    def __init__(self, path):
        super(CovarianceH5, self).__init__(path)
        self.array_data = DataSet(Covariance.array_data, expand_dimension=2)
        self.source = Reference(Covariance.source)
        self._end_accessor_declarations()

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.array_data.append(partial_result, close_file=False)


class CorrelationCoefficientsH5(H5File):

    def __init__(self, path):
        super(CorrelationCoefficientsH5, self).__init__(path)
        self.array_data = DataSet(CorrelationCoefficients.array_data)
        self.source = Reference(CorrelationCoefficients.source)
        self.labels_ordering = Json(CorrelationCoefficients.labels_ordering)
        self._end_accessor_declarations()


class ConnectivityMeasureH5(H5File):

    def __init__(self, path):
        super(ConnectivityMeasureH5, self).__init__(path)
        self.connectivity = Reference(ConnectivityMeasure.connectivity)
        self._end_accessor_declarations()
