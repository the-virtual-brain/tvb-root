from tvb.basic.neotraits.api import HasTraits, Attr, NArray


class BazDataType(HasTraits):
    miu = NArray()
    scalar_str = Attr(str)


class FooDatatype(HasTraits):
    array_float = NArray()
    array_int = NArray(dtype=int, ndim=2)
    scalar_int = Attr(int)
    abaz = Attr(field_type=BazDataType)
    some_transient = NArray(ndim=3, required=False)


class BarDatatype(FooDatatype):
    array_str = NArray(dtype='S32', ndim=1)

