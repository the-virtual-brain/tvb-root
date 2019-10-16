from tvb.basic.neotraits.api import HasTraits, Attr, NArray


class FooDatatype(HasTraits):
    array_float = NArray()
    array_int = NArray(dtype=int, ndim=2)
    scalar_str = Attr(str)
    scalar_int = Attr(int)
    non_mapped_attr = NArray()



