from sqlalchemy import ForeignKey, Integer, Column

from tvb.core.entities.model.model_datatype import DataType


class SimulationStateIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
