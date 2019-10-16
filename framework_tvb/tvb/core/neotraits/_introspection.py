"""
common introspection code
"""
import typing
from tvb.basic.neotraits.api import HasTraits, Attr

if typing.TYPE_CHECKING:
    from tvb.basic.neotraits._core import MetaType


class DeclarativeFieldsTypeMixin(type):

    def gather_declared_fields(cls):
        # type: (type) -> typing.Sequence[Attr]
        """
        If the type cls does not have a attribute trait of type HasTraits this does nothing.
        If trait is declared but no fields, then return all the declared fields in that trait,
        except the ones inherited from a superclass.
        If there is also a fields attribute then only check that it contains attributes that are declared on the trait.
        """
        if not hasattr(cls, 'trait'):
            return []

        trait = getattr(cls, 'trait')  # type: MetaType
        if not isinstance(trait, type) or not issubclass(trait, HasTraits):
            raise AttributeError('trait attribute is required to be a HasTraits')


        if 'fields' not in cls.__dict__:
            # no hasattr because we do not want to inherit the fields attribute
            # When fields is not given default to all own_declarative_attrs
            return [getattr(trait, fn) for fn in trait.own_declarative_attrs]

        all_declared_attrs = [getattr(trait, fn) for fn in trait.declarative_attrs]

        fields = []
        for f in cls.__dict__['fields']:
            # Fields are explicitly given, so we allow fields from the superclass to be present
            if isinstance(f, str) and f in trait.declarative_attrs:
                fields.append(getattr(trait, f))
            elif f in all_declared_attrs:
                fields.append(f)
            else:
                raise ValueError('fields should contain either the names of '
                                     'the traited fields or the fields themselves')
        return fields

    @property
    def all_declared_fields(cls):
        """
            merge all field declarations from the type hierarchy
            """
        ret = []
        for super_cls in cls.mro():
            for attr_name in getattr(super_cls, 'fields', []):
                if attr_name not in ret:  # attr was overridden, don't duplicate
                    ret.append(attr_name)
        return tuple(ret)
