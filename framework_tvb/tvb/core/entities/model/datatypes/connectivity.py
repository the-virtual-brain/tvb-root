from __future__ import absolute_import

from sqlalchemy import Column, Integer, ForeignKey, Boolean, Float

from tvb.core.neotraits.db import from_ndarray

from ..model_datatype import DataType


class ConnectivityIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    number_of_regions = Column(Integer, nullable=False)
    number_of_connections = Column(Integer, nullable=False)
    undirected = Column(Boolean)

    weights_min = Column(Float)
    weights_max = Column(Float)
    weights_mean = Column(Float)

    tract_lengths_min = Column(Float)
    tract_lengths_max = Column(Float)
    tract_lengths_mean = Column(Float)

    #TODO: keep these metadata?
    # weights_non_zero_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    # weights_non_zero = relationship(NArrayIndex, foreign_keys=weights_non_zero_id, primaryjoin=NArrayIndex.id == weights_non_zero_id)

    # tract_lengths_non_zero_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    # tract_lengths_non_zero = relationship(NArrayIndex, foreign_keys=tract_lengths_non_zero_id, primaryjoin=NArrayIndex.id == tract_lengths_non_zero_id)
    #
    # tract_lengths_connections_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    # tract_lengths_connections = relationship(NArrayIndex, foreign_keys=tract_lengths_connections_id, primaryjoin=NArrayIndex.id == tract_lengths_connections_id)

    def fill_from_has_traits(self, datatype):
        self.number_of_regions = datatype.number_of_regions
        self.number_of_connections = datatype.number_of_connections
        self.undirected = datatype.undirected
        self.weights_min, self.weights_max, self.weights_mean = from_ndarray(datatype.weights)
        self.tract_lengths_min, self.tract_lengths_max, self.tract_lengths_mean = from_ndarray(datatype.tract_lengths)
        # self.weights_non_zero = NArrayIndex.from_ndarray(datatype.weights[datatype.weights.nonzero()])
        # self.tract_lengths_non_zero = NArrayIndex.from_ndarray(datatype.tract_lengths[datatype.tract_lengths.nonzero()])
        # self.tract_lengths_connections = NArrayIndex.from_ndarray(datatype.tract_lengths[datatype.weights.nonzero()])
