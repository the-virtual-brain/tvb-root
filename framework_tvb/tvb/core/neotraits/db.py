# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import uuid
from datetime import datetime
import numpy
from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy import String, Boolean
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
    create_date = Column(DateTime, default=datetime.now)

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
        self.gid = uuid.uuid4().hex

    def __repr__(self):
        cls = type(self)
        return '<{}.{} gid="{}..." id="{}">'.format(
            cls.__module__, cls.__name__, self.gid[:4], self.id
        )


def ensure_float(data):
    return numpy.float64(data)


def ensure_int(data):
    return int(data)


def from_ndarray(array):
    if array is None:
        return None

    if array.dtype.kind in 'iufc' and array.size != 0:
        # we compute these simple statistics for integer unsigned float or complex
        # arrays that are not empty
        minvalue, maxvalue = ensure_float(array.min()), ensure_float(array.max())
        median = numpy.median(array)
    else:
        minvalue, maxvalue, median = None, None, None

    return minvalue, maxvalue, median


def prepare_array_shape_meta(shape_array):
    length_1d = None
    length_2d = None
    length_3d = None
    length_4d = None

    try:
        length_1d = shape_array[0]
        length_2d = shape_array[1]
        length_3d = shape_array[2]
        length_4d = shape_array[3]
    except IndexError:
        pass

    shape = []
    for length in [length_1d, length_2d, length_3d, length_4d]:
        if length:
            shape.append(length)
    return tuple(shape)