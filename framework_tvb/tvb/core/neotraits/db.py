from sqlalchemy import Column, Integer
from sqlalchemy import String, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta, declared_attr
from sqlalchemy.orm import relationship
from tvb.basic.neotraits.api import Attr, NArray
from ._introspection import gather_declared_fields


class DeclarativeTraitMeta(DeclarativeMeta):
    """
    Extends the sqlalchemy declarative metaclass by recognizing the trait/fields declarations
    and autogenerating Columns for them.
    This has to happen early, in a metaclass, so that sqlalchemy gets to see the new fields.
    Sqlalchemy idiomatically does the introspection for models early, at class creation time.
    """
    def __new__(mcs, type_name, bases, namespace):
        if 'trait' not in namespace:
            return super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)

        # It seems that sqlalchemy tolerates setting columns after the type has been created.
        # We take advantage of this to reuse the gather_declared_fields

        cls = super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)  # type: DeclarativeTraitMeta
        if 'trait' not in namespace:
            # huh! cause sqlalchemy uses this meta to make some other objects
            return cls

        fields = gather_declared_fields(cls)
        cls.fields = fields

        for f in fields:
            cls._generate_columns_for_traited_attribute(f)

        return cls

    def _generate_columns_for_traited_attribute(cls, f):
        # type: (Attr) -> None
        if isinstance(f, NArray):
            narray_fk_column = Column(Integer, ForeignKey('narrays.id'), nullable=not f.required)
            setattr(cls, f.field_name + '_id', narray_fk_column)
            setattr(cls, f.field_name, relationship(NArrayIndex, foreign_keys=narray_fk_column))
        elif isinstance(f, Attr):
            # todo this in a sane way
            if f.field_type == bool:
                setattr(cls, f.field_name, Column(Boolean, nullable=not f.required))
            elif f.field_type == int:
                setattr(cls, f.field_name, Column(Integer, nullable=not f.required))
            elif f.field_type == str:
                setattr(cls, f.field_name, Column(String, nullable=not f.required))
            else:
                raise NotImplementedError()
        else:
            raise AttributeError(
                "fields must contain declarative "
                "attributes from a traited class not a {}".format(type(f)))

# well sqlalchemy is not so happy with 2 declarative bases
# so if we dance with metaclasses then it is one
# don't quite like this
# If we make a function that creates the fields automatically it still needs to call a metaclass to build
# the damn class. So that approach is almost the same


Base = declarative_base(name='DeclarativeBase', metaclass=DeclarativeTraitMeta)


class HasTraitsIndex(Base):
    id = Column(Integer, primary_key=True)
    gid = Column(String(32), unique=True)
    type_ = Column(String(50))

    @declared_attr
    def __tablename__(cls):
        return cls.__name__

    __mapper_args__ = {
        'polymorphic_on': type_,
        'polymorphic_identity': __tablename__
    }

    def __init__(self):
        super(HasTraitsIndex, self).__init__()
        self.type_ = type(self).__name__

    def fill_from_has_traits(self, datatype):
        cls = type(self)
        if not hasattr(cls, 'trait'):
            raise NotImplementedError(
                'the default implementation works only if an associated '
                'datatype has been declared by setting the trait field')

        trait = getattr(cls, 'trait')
        if not isinstance(datatype, trait):
            raise TypeError('datatype must be of the declared type {} not {}'.format(trait, type(datatype)))

        self.gid = datatype.gid.hex

        for f in cls.fields:
            if isinstance(f, NArray):
                field_value = getattr(datatype, f.field_name)
                arr = NArrayIndex(
                    dtype=str(field_value.dtype),
                    ndim=field_value.ndim,
                    shape=str(field_value.shape),
                    dim_names=str(f.dim_names),
                    minvalue=field_value.min(),
                    maxvalue=field_value.max())
                setattr(self, f.field_name, arr)
            else:
                # todo: handle non-array nonscalar Attr's
                scalar = getattr(datatype, f.field_name)
                setattr(self, f.field_name, scalar)



class NArrayIndex(Base):
    __tablename__ = 'narrays'

    id = Column(Integer, primary_key=True)
    dtype = Column(String(64), nullable=False)
    ndim = Column(Integer, nullable=False)
    shape = Column(String, nullable=False)
    dim_names = Column(String)
    minvalue = Column(Float)
    maxvalue = Column(Float)
