import json
import numpy
from sqlalchemy import Column, Integer, Text
from sqlalchemy import String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base, declared_attr

SCALAR_MAPPING = {
    bool: Boolean,
    int: Integer,
    str: String
}


Base = declarative_base(name='DeclarativeBase')


class HasTraitsIndex(Base):
    id = Column(Integer, primary_key=True)
    gid = Column(String(32), unique=True)
    type_ = Column(String(50))
    title = Column(Text)

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

    def __repr__(self):
        cls = type(self)
        return '<{}.{} gid="{}..." id="{}">'.format(
            cls.__module__, cls.__name__, self.gid[:4], self.id
        )


class NArrayIndex(Base):
    __tablename__ = 'narrays'

    id = Column(Integer, primary_key=True)
    _dtype = Column(String(64), nullable=False)
    ndim = Column(Integer, nullable=False)
    _shape = Column(Text, nullable=False)
    _dim_names = Column(Text)

    has_nan = Column(Boolean, nullable=False, default=False)
    # some statistics, null if they make no sense for the dtype
    min_value = Column(Float)
    max_value = Column(Float)
    median_value = Column(Float)

    # unrolled shape for easy querying
    length_1d = Column(Integer)
    length_2d = Column(Integer)
    length_3d = Column(Integer)
    length_4d = Column(Integer)

    @property
    def dtype(self):
        # this complex serialisation of datatypes is to support complex datatypes
        return numpy.dtype([tuple(i) for i in json.loads(self._dtype)])

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = json.dumps(dtype.descr)

    @property
    def shape(self):
        return tuple(json.loads(self._shape))

    @shape.setter
    def shape(self, val):
        self._shape = json.dumps(val)

    @property
    def dim_names(self):
        return json.loads(self._dim_names)

    @dim_names.setter
    def dim_names(self, val):
        self._dim_names = json.dumps(val)

    @classmethod
    def from_ndarray(cls, array):
        if array.dtype.kind in 'iufc' and array.size != 0:
            # we compute these simple statistics for integer unsigned float or complex
            # arrays that are not empty
            has_nan = numpy.isnan(array).any()
            minvalue, maxvalue = array.min(), array.max()
            median = numpy.median(array)
        else:
            has_nan = False
            minvalue, maxvalue, median = None, None, None

        self = cls(
            ndim=array.ndim,
            has_nan=has_nan,
            min_value=minvalue,
            max_value=maxvalue,
            median_value=median
        )

        self.dtype = array.dtype
        self.shape = array.shape

        for i, l in enumerate(array.shape):
            if i > 4:
                break
            setattr(self, 'length_{}d'.format(i + 1), l)

        return self

    def __repr__(self):
        cls = type(self)
        return '<{}.{} id="{}" dtype="{}", shape="{}">'.format(
            cls.__module__, cls.__name__, self.id, self.dtype, self.shape
        )
