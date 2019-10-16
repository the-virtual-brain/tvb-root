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

        cls = super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)
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
            narray_fk_column = Column(Integer, ForeignKey('narrays.id'), nullable=False)
            setattr(cls, f.field_name + '_id', narray_fk_column)
            setattr(cls, f.field_name, relationship(NArrayIndex, foreign_keys=narray_fk_column))
        elif isinstance(f, Attr):
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

# well sqlalchemy is not so happy with 2 declarative bases
# so if we dance with metaclasses then it is one
# don't quite like this
# If we make a function that creates the fields automatically it still needs to call a metaclass to build
# the damn class. So that approach is almost the same


class _Base(object):
    """
    A base class for all db entities.
    This is a sqlalchemy thing.
    You can't simply inherit the class returned by declarative_base
    It will try to map your abstract base to the db.
    It is after all a sqlalchemy model if it inherits the declarative base.
    So we make this simple base and tell declarative_base to insert this in the mro of it's returned base.
    __abstract__ seems to work too. So which one is the preferred way?
    """
    id = Column(Integer, primary_key=True)

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    def from_datatype(self, datatype):
        if not hasattr(type(self), 'trait'):
            raise NotImplementedError(
                'the default implementation works only if an associated '
                'datatype has been declared by setting the trait field')

        trait = getattr(type(self), 'trait')
        if not isinstance(datatype, trait):
            raise TypeError('datatype must be of the declared type {} not {}'.format(trait, type(datatype)))

        for f in self.fields:
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


Base = declarative_base(name='DeclarativeBase', cls=_Base, metaclass=DeclarativeTraitMeta)


class NArrayIndex(Base):
    __tablename__ = 'narrays'

    id = Column(Integer, primary_key=True)
    dtype = Column(String(64), nullable=False)
    ndim = Column(Integer, nullable=False)
    shape = Column(String, nullable=False)
    dim_names = Column(String)
    minvalue = Column(Float)
    maxvalue = Column(Float)


