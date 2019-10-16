from sqlalchemy import Column, Integer
from sqlalchemy import String, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import relationship
from tvb.basic.neotraits import api as neo
from ._introspection import gather_declared_fields


class DeclarativeTraitMeta(DeclarativeMeta):
    def __new__(mcs, type_name, bases, namespace):
        if 'trait' not in namespace:
            return super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)

        namespace['id'] = Column(Integer, primary_key=True)
        if '__tablename__' not in namespace:
            namespace['__tablename__'] = type_name

        # It seems that sqlalchemy tolerates setting columns after the type has been created.
        # We take advantage of this to reuse the gather_declared_fields

        cls = super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)
        if 'trait' not in namespace:
            # huh! cause sqlalchemy uses this meta to make some other objects
            return cls

        fields = gather_declared_fields(cls)

        for f in fields:
            if isinstance(f, neo.NArray):
                narray_fk_column = Column(Integer, ForeignKey('narrays.id'), nullable=False)
                setattr(cls, f.field_name + '_id', narray_fk_column)
                setattr(cls, f.field_name, relationship(NArray, foreign_keys=narray_fk_column))
            elif isinstance(f, neo.Attr):
                # todo this in a sane way
                if f.field_type == bool:
                    setattr(cls, f.field_name, Column(Boolean, nullable=False))
                elif f.field_type == int:
                    setattr(cls, f.field_name, Column(Integer, nullable=False))
                elif f.field_type == str:
                    setattr(cls, f.field_name, Column(String, nullable=False))
                else:
                    raise NotImplementedError()
            else:
                raise AttributeError("fields must contain declarative attributes from a traited class not a {}".format(type(f)))

        return cls

# well sqlalchemy is not so happy with 2 declarative bases
# so if we dance with metaclasses then it is one
# don't quite like this
# If we make a function that creates the fields automatically it still needs to call a metaclass to build
# the damn class. So that approach is almost the same


Base = declarative_base(name='Base', metaclass=DeclarativeTraitMeta)



class NArray(Base):
    __tablename__ = 'narrays'

    id = Column(Integer, primary_key=True)
    dtype = Column(String(64), nullable=False)
    ndim = Column(Integer, nullable=False)
    shape = Column(String, nullable=False)
    dim_names = Column(String)
    minvalue = Column(Float)
    maxvalue = Column(Float)


