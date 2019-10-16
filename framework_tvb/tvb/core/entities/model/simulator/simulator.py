from sqlalchemy import Column, Integer, ForeignKey

from tvb.core.entities.model.model_datatype import DataType


class SimulatorIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
