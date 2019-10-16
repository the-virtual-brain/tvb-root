from sqlalchemy import Column, Integer
from sqlalchemy import String, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta, declared_attr
from sqlalchemy.orm import relationship
from tvb.basic.neotraits.api import Attr, NArray, HasTraits
from ._introspection import DeclarativeFieldsTypeMixin

SCALAR_MAPPING = {
    bool: Boolean,
    int: Integer,
    str: String
}


class DeclarativeTraitMeta(DeclarativeFieldsTypeMixin, DeclarativeMeta):
    """
    Extends the sqlalchemy declarative metaclass by recognizing the trait/fields declarations
    and autogenerating Columns for them.
    This has to happen early, in a metaclass, so that sqlalchemy gets to see the new fields.
    Sqlalchemy idiomatically does the introspection for models early, at class creation time.
    """
    def __new__(mcs, type_name, bases, namespace):
        # It seems that sqlalchemy tolerates setting columns after the type has been created.
        # We take advantage of this to reuse the gather_declared_fields

        cls = super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)  # type: DeclarativeTraitMeta
        if 'trait' not in namespace:
            # huh! cause sqlalchemy uses this meta to make some other objects
            return cls

        fields = cls.gather_declared_fields()
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
            if issubclass(f.field_type, HasTraits):
                # setattr(cls, f.field_name, Column(Integer, ForeignKey(HasTraitsIndex.id), nullable=not f.required))
                # a fk on the supertype HasTraitsIndex is not correct
                # an automatic mapping of these is problematic. First we don't force a canonic global mapping,
                # we don't autogen all mapping classes, so there is no way to know that this attr's index type is
                pass
            else:
                sqlalchemy_type = SCALAR_MAPPING.get(f.field_type)
                if not sqlalchemy_type:
                    raise NotImplementedError()
                setattr(cls, f.field_name, Column(sqlalchemy_type, nullable=not f.required))
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

    # Quick remainder about @declared_attr. It makes a class method.
    # Sqlalchemy will treat this class method like a statically declared class attribute
    # Another quick remainder, class methods are polymorphic

    @declared_attr
    def __tablename__(cls):
        # subclasses no longer need to define the __tablename__ as we do it here polymorphically for them.
        return cls.__name__

    @declared_attr
    def __mapper_args__(cls):
        """
        A polymorphic __maper_args__. Because of it the subclasses no longer need to declare the polymorphic_identity
        """
        # this gets called by sqlalchemy before the HasTraitsIndex class declaration is finished (in it's metatype)
        # so we have to refer to the type by name
        if cls.__name__ == 'HasTraitsIndex' and cls.__module__ == __name__:
            # for the base class we have to define the discriminator column
            return {
                'polymorphic_on': cls.type_,
                'polymorphic_identity': cls.__name__
            }
        else:
            return {
                'polymorphic_identity': cls.__name__
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

        for f in cls.all_declared_fields:
            field_value = getattr(datatype, f.field_name)
            if field_value is None:
                setattr(self, f.field_name, None)
            elif isinstance(f, NArray):
                try:
                    minvalue, maxvalue = field_value.min(), field_value.max()
                except TypeError:
                    # dtype is string or other non comparable type
                    minvalue, maxvalue = None, None

                arr = NArrayIndex(
                    dtype=str(field_value.dtype),
                    ndim=field_value.ndim,
                    shape=str(field_value.shape),
                    dim_names=str(f.dim_names),
                    minvalue=minvalue,
                    maxvalue=maxvalue
                )
                setattr(self, f.field_name, arr)
            elif isinstance(f, Attr):
                if f.field_type in SCALAR_MAPPING:
                    setattr(self, f.field_name, field_value)
                elif issubclass(f.field_type, HasTraits):
                    # the user has to deal with trait db composition manually
                    pass
                else:
                    raise NotImplementedError("don't know how to map {}".format(f))



class NArrayIndex(Base):
    __tablename__ = 'narrays'

    id = Column(Integer, primary_key=True)
    dtype = Column(String(64), nullable=False)
    ndim = Column(Integer, nullable=False)
    shape = Column(String, nullable=False)
    dim_names = Column(String)
    minvalue = Column(Float)
    maxvalue = Column(Float)
