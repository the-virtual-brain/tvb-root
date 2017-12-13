# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
"""
Cascaded inheritance test.

As we want to make this as realistic as possible, we will use DATATYPE table name.
As a direct consequence, we can not use this test with the rest of the tests, 
unless we rewrite major parts from the setup.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker
#from sqlalchemy.ext.declarative import declarative_base
from tvb.basic.traits.core import TypeBase as BASE
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile


#BASE = declarative_base()
ENGINE = create_engine(TvbProfile.current.db.DB_URL, pool_recycle=5, echo=True)
SESSION = sessionmaker(bind=ENGINE)
LOGGER = get_logger(__name__)


class Type(object):
    pass

class DataType(BASE):
    
    id = Column(Integer, primary_key=True)
    granpda_name = Column(String)
    __tablename__ = "DATA_TYPES"

class MappedType(DataType, Type):
    __tablename__ = None

class MappedArrayData(MappedType):
    """A Person"""
    id = Column('id', Integer, ForeignKey(DataType.id, ondelete="CASCADE"), primary_key=True)
    parent_name = Column(String)
    __tablename__ = "MAPPED_ARRAY"
    #__mapper_args__ = {'inherit_condition': id == DataType.id}
    
class MappedArray(MappedArrayData):
    __tablename__ = None
    ### Won't be saved, because it does not fit into the SQL's patterns
    extra_field = Column(String)
 
class ConnectivityMeasureData(MappedArray):
    """A Person"""
    id = Column('id', Integer, ForeignKey(MappedArray.id, ondelete="CASCADE"), primary_key=True)
    child_name = Column(String) 
    __tablename__ = "CONN_MEASURE"
    #__mapper_args__ = {'inherit_condition': id == MappedArrayData.id}  
    
 
class ConnectivityMeasure(ConnectivityMeasureData): 
    __tablename__ = None
    
    
###### SECOND TEST -- INHERITRANCE AND RETRIEVAL #########

class PolyBaseBase(BASE):
    
    id = Column(Integer, primary_key=True)
    object_name = Column(String)
    __tablename__ = "POLY_BASE_BASE"
    
    
class PolymorphicBase(PolyBaseBase):
    id = Column('id', Integer, ForeignKey(PolyBaseBase.id, ondelete="CASCADE"), primary_key=True)
    surface_type = Column(String)
    #object_name = Column(String)
    __tablename__ = "POLY_BASE"
    __mapper_args__ = {'polymorphic_on': surface_type, 
                       'polymorphic_identity': 'BASE',
                       'exclude_properties': None,
                       'inherit_condition': (id == PolyBaseBase.id)}

class PolymorphicSubClass(PolymorphicBase):
    child_field = Column(String)
    ### Doesn't work to store value on this field not even with standard SqlAlchemy!!!
    ### DB Column will get created on the "POLY_BASE" table, but values are not stored.
    __mapper_args__ = {'polymorphic_identity': 'CHILD',
                       'exclude_properties': None}
    
    
class PolymorphicFramework(PolymorphicSubClass):
    __tablename__ = None
    __mapper_args__ = {'polymorphic_identity': 'CHILD'}
    
class PolymorphicBase2(PolymorphicBase):
    __tablename__ = None
    #surface_type2 = column_property(case([(literal(True), "BASE"), ], else_="BASE"))
    #surface_type = text("BASE")
    #surface_type2 = column_property(Column(String))
    __mapper_args__ = {'polymorphic_identity': '*'}
#    surface_type2 = Column(String)
#    __mapper_args__ = {"polymorphic_on":case([(surface_type2 == "EN", "BASE1"), (surface_type2 == "MA", "BASE2"), ], else_="BASE"),
#                       "polymorphic_identity":"BASE" }
    
    
class DBMappingTest():
    
    
    def test_storage(self):
        session = SESSION()
        BASE.metadata.create_all(bind = session.connection())
        session.commit()
        session.close()
        
        child = ConnectivityMeasure()
        child.child_name = "Child Name 43"
        child.extra_field = "From table not mapped"
        child.parent_name = "Parent Name 43"
        child.granpda_name = "Granpa Name 43"
        session = SESSION()
        session.add(child)
        session.commit()
        
        session = SESSION()
        retrieved_child = session.query(ConnectivityMeasure).filter_by(id=child.id).one()
        assert child.granpda_name == retrieved_child.granpda_name
        assert child.child_name == retrieved_child.child_name
        #assert child.extra_field, retrieved_child.extra_field)
        assert child.parent_name == retrieved_child.parent_name
        session.close()
    
    
    def test_inheritance(self):
        session = SESSION()
        BASE.metadata.create_all(bind = session.connection())
        all_class = session.query(PolymorphicBase).all()
        for entity in all_class:
            session.delete(entity)
        session.commit()
        session.close()
        session.commit()
        session.close()
        
        parent = PolymorphicBase()
        parent.surface_type = "BASE"
        parent.object_name = "grandpa"
        session.add(parent)
        child = PolymorphicSubClass()
        child.object_name = "grandpa"
        child.surface_type = "CHILD"
        child.child_field = "test-child"
        session.add(child)
        session.commit()
        
        session = SESSION()
        all_class = session.query(PolymorphicBase).all()
        assert 2 == len(all_class)
        assert "grandpa" == all_class[0].object_name
        all_subclasses = session.query(PolymorphicSubClass).all()
        assert 1 == len(all_subclasses)
        assert "grandpa" == all_subclasses[0].object_name
        assert "test-child" == all_subclasses[0].child_field
        
        all_subclasses = session.query(PolymorphicFramework).all()
        assert 1 == len(all_subclasses)
        all_class = session.query(PolymorphicBase2).all()
        #assert 2, len(all_class))
        session.close()
        
        session = SESSION()
        all_class = session.query(PolymorphicBase).all()
        for entity in all_class:
            session.delete(entity)
        session.commit()
        session.close()
