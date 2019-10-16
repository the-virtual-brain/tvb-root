import scipy.sparse
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from tvb.datatypes.local_connectivity import LocalConnectivity

from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.neotraits.db import HasTraitsIndex, NArrayIndex


class LocalConnectivityIndex(HasTraitsIndex):
    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)

    surface_id = Column(Integer, ForeignKey(SurfaceIndex.id), nullable=not LocalConnectivity.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_id, primaryjoin=SurfaceIndex.id == surface_id)

    matrix_non_zero_id = Column(Integer, ForeignKey('narrays.id'), nullable=False)
    matrix_non_zero = relationship(NArrayIndex, foreign_keys=matrix_non_zero_id)

    def fill_from_has_traits(self, datatype):
        self.gid = datatype.gid.hex
        I, J, V = scipy.sparse.find(datatype.matrx)
        self.matrix_non_zero = NArrayIndex.from_ndarray(V)
