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



class NArrayIndex(Base):
    __tablename__ = 'narrays'

    id = Column(Integer, primary_key=True)
    dtype = Column(String(64), nullable=False)
    ndim = Column(Integer, nullable=False)
    shape = Column(Text, nullable=False)
    dim_names = Column(Text)
    minvalue = Column(Float)
    maxvalue = Column(Float)
    # unrolled shape for easy querying
    length_1d = Column(Integer)
    length_2d = Column(Integer)
    length_3d = Column(Integer)
    length_4d = Column(Integer)

    @classmethod
    def from_ndarray(cls, array):
        try:
            minvalue, maxvalue = array.min(), array.max()
        except TypeError:
            # dtype is string or other non comparable type
            minvalue, maxvalue = None, None

        self = cls(
            dtype=str(array.dtype),
            ndim=array.ndim,
            shape=str(array.shape),
            minvalue=minvalue,
            maxvalue=maxvalue
        )

        for i, l in enumerate(array.shape):
            if i > 4:
                break
            setattr(self, 'length_{}d'.format(i + 1), l)

        return self

