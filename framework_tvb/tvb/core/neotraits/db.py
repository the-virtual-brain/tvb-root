from sqlalchemy import Column, Integer
from sqlalchemy import String, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import relationship
from tvb.basic.neotraits import api as neo


class DeclarativeTraitMeta(DeclarativeMeta):
    def __new__(mcs, type_name, bases, namespace):
        if 'trait' not in namespace:
            # huh! cause sqlalchemy uses this meta to make some other objects
            return super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)

        trait = namespace.get('trait')
        if not isinstance(trait, type) or not issubclass(trait, neo.HasTraits):
            raise AttributeError('trait attribute is required to be a HasTraits')

        all_fields = [getattr(trait, fn) for fn in trait.own_declarative_attrs]

        if 'fields' not in namespace:
            fields = all_fields
        else:
            fields = []
            for f in namespace['fields']:
                if isinstance(f, str) and f in trait.own_declarative_attrs:
                    fields.append(getattr(trait, f))
                elif f in all_fields:
                    fields.append(f)
                else:
                    raise ValueError('fields should contain either the names of '
                                     'the traited fields or the fields themselves')

        namespace['id'] = Column(Integer, primary_key=True)

        for f in fields:
            if isinstance(f, neo.NArray):
                narray_fk_column = Column(Integer, ForeignKey('narrays.id'), nullable=False)
                namespace[f.field_name + '_id'] = narray_fk_column
                namespace[f.field_name] = relationship(NArray, foreign_keys=narray_fk_column)
            elif isinstance(f, neo.Attr):
                # todo this in a sane way
                if f.field_type == bool:
                    namespace[f.field_name] = Column(Boolean, nullable=False)
                elif f.field_type == int:
                    namespace[f.field_name] = Column(Integer, nullable=False)
                elif f.field_type == str:
                    namespace[f.field_name] = Column(String, nullable=False)
                else:
                    raise NotImplementedError()
            else:
                raise AttributeError("fields must contain declarative attributes from a traited class not a {}".format(type(f)))

        if '__tablename__' not in namespace:
            namespace['__tablename__'] = type_name

        cls = super(DeclarativeTraitMeta, mcs).__new__(mcs, type_name, bases, namespace)
        # it seems that sqlalchemy tolerates setting column after the type has been created
        # we could have done the thing here too
        return cls

# well sqlalchemy is not so happy with 2 declarative bases
# so if we dance with metaclasses then it is one
# don't quite like this
# If we make a function that creates the fields automatically it still needs to call a metaclass to build
# the damn class. So that approach is almost the same
# Base = declarative_base()


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


