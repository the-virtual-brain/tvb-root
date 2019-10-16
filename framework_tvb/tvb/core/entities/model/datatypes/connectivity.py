from sqlalchemy import Column, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from tvb.core.neotraits.db import HasTraitsIndex, NArrayIndex


class ConnectivityIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    number_of_regions = Column(Integer, nullable=False)
    number_of_connections = Column(Integer, nullable=False)
    undirected = Column(Boolean)

    areas_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    areas = relationship(NArrayIndex, foreign_keys=areas_id)

    weights_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    weights = relationship(NArrayIndex, foreign_keys=weights_id)

    weights_non_zero_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    weights_non_zero = relationship(NArrayIndex, foreign_keys=weights_non_zero_id)

    tract_lengths_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    tract_lengths = relationship(NArrayIndex, foreign_keys=tract_lengths_id)

    tract_lengths_non_zero_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    tract_lengths_non_zero = relationship(NArrayIndex, foreign_keys=tract_lengths_non_zero_id)

    tract_lengths_connections_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    tract_lengths_connections = relationship(NArrayIndex, foreign_keys=tract_lengths_connections_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        self.number_of_regions = datatype.number_of_regions
        self.number_of_connections = datatype.number_of_connections
        self.undirected = datatype.undirected
        self.areas = NArrayIndex.from_ndarray(datatype.areas)
        self.weights = NArrayIndex.from_ndarray(datatype.weights)
        self.weights_non_zero = NArrayIndex.from_ndarray(datatype.weights[datatype.weights.nonzero()])
        self.tract_lengths = NArrayIndex.from_ndarray(datatype.tract_lengths)
        self.tract_lengths_non_zero = NArrayIndex.from_ndarray(datatype.tract_lengths[datatype.tract_lengths.nonzero()])
        self.tract_lengths_connections = NArrayIndex.from_ndarray(datatype.tract_lengths[datatype.weights.nonzero()])
