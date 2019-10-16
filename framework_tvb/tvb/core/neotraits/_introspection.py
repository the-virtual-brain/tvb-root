"""
common introspection code
"""
import typing
from tvb.basic.neotraits.api import HasTraits, Attr

if typing.TYPE_CHECKING:
    from tvb.basic.neotraits._core import MetaType


def gather_declared_fields(cls):
    # type: (type) -> typing.Sequence[Attr]
    """
    If the type cls does not have a attribute trait of type HasTraits this does nothing.
    If there is a trait then return all the declared fields.
    If there is also a fields attribute then only check that it contains attributes that are declared on the trait.
    """
    if not hasattr(cls, 'trait'):
        return []

    trait = getattr(cls, 'trait')  # type: MetaType
    if not isinstance(trait, type) or not issubclass(trait, HasTraits):
        raise AttributeError('trait attribute is required to be a HasTraits')

    all_fields = [getattr(trait, fn) for fn in trait.declarative_attrs]

    fields = []

    for base in cls.mro():
        for f in getattr(base, 'fields', []):
            if isinstance(f, str) and f in trait.declarative_attrs:
                fields.append(getattr(trait, f))
            elif f in all_fields:
                fields.append(f)
            else:
                raise ValueError('fields should contain either the names of '
                                 'the traited fields or the fields themselves')
    if fields:
        return fields
    else:
        return all_fields
