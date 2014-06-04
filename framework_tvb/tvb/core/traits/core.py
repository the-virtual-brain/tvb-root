# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <mw@eml.cc>
"""

import re
import sqlalchemy
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from tvb.core.traits.sql_mapping import get_sql_mapping
from tvb.basic.traits.core import MetaType, Type, SPECIAL_KWDS, KWARS_USE_STORAGE
from tvb.basic.logger.builder import get_logger


LOG = get_logger(__name__)

SPECIAL_KWDS.remove(KWARS_USE_STORAGE)


def compute_table_name(class_name):
    """
    Given a class name compute the name of the corresponding SQL table.
    """
    tablename = 'MAPPED' + re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', '_', class_name).upper()
    if tablename.count('MAPPED_') > 1:
        tablename = tablename.replace('MAPPED_', '', 1)
    return tablename



class DeclarativeMetaType(DeclarativeMeta, MetaType):
    """
    The DeclarativeMetaType class helps with class creation by automating
    some of the sqlalchemy code generation. We code for three possibilities:

    - the sql or db keywords are False, no sqlalch used
    - sql or db keywords are True or unset, default sqlalche used
    - sql or db keywords are set with sqlalchemy.Column instances, and
        that is used

    If it is desired that no sql/db is used, import traits.core and set
    TVBSettings.TRAITS_CONFIGURATION.use_storage = False. This will have the (hopefully
    desired) effect that all sql and db keyword args are ignored.

    """

    def __new__(*args):
        mcs, name, bases, dikt = args
        if dikt.get('__generate_table__', False):
            tablename = compute_table_name(name)
            if '__tablename__' not in dikt:
                dikt['__tablename__'] = tablename
        newcls = super(DeclarativeMetaType, mcs).__new__(*args)

        if newcls.__name__ in ('DataType', 'MappedType'):
            return newcls

        mro_names = map(lambda cls: cls.__name__, newcls.mro())
        if Type in newcls.mro() and 'DataType' in mro_names:
            LOG.debug('new mapped, typed class %r', newcls)
        else:
            LOG.debug('new mapped, non-typed class %r', newcls)
            return newcls

        ## Compute id foreign-key to parent
        all_parents = []
        for b in bases:
            all_parents.extend(b.mro())
        mapped_parent = filter(lambda cls: issubclass(cls, Type) and hasattr(cls, '__tablename__')
                               and getattr(cls, '__tablename__') is not None, all_parents)
        # Identify DATA_TYPE class, to be used for specific references
        datatype_class = filter(lambda cls: hasattr(cls, '__tablename__') and cls.__tablename__ == 'DATA_TYPES', 
                                all_parents)[0]

        ###### Map Trait attributes to SQL Columns as necessary
        all_class_traits = getattr(newcls, 'trait', {})
        super_traits = dict()
        for parent_class in filter(lambda cls: issubclass(cls, Type), all_parents):
            super_traits.update(getattr(parent_class, 'trait', {}))
        newclass_only_traits = dict([(key, all_class_traits[key])
                                     for key in all_class_traits if key not in super_traits])

        LOG.debug('mapped, typed class has traits %r', newclass_only_traits)
        for key, attr in newclass_only_traits.iteritems():
            kwd = attr.trait.inits.kwd
            ##### Either True or a Column instance
            sql = kwd.get('db', True)

            if isinstance(sql, sqlalchemy.Column):
                setattr(newcls, '_' + key, sql)

            elif get_sql_mapping(attr.__class__):
                defsql = get_sql_mapping(attr.__class__)
                sqltype, args, kwds = defsql[0], (), {}
                for arg in defsql[1:]:
                    if type(arg) is tuple:
                        args = arg
                    elif type(arg) is dict:
                        kwds = arg
                setattr(newcls, '_' + key, sqlalchemy.Column('_' + key, sqltype, *args, **kwds))

            elif Type in attr.__class__.mro() and hasattr(attr.__class__, 'gid'):
                #### Is MappedType
                fk = sqlalchemy.ForeignKey('DATA_TYPES.gid', ondelete="SET NULL")
                setattr(newcls, '_' + key, sqlalchemy.Column('_' + key, sqlalchemy.String, fk))
                if newcls.__tablename__:
                    #### Add relationship for specific class, to have the original entity loaded
                    #### In case of cascade = 'save-update' we would need to SET the exact instance type
                    #### as defined in atrr description
                    rel = relationship(attr.__class__, lazy='joined', cascade="none",
                                       primaryjoin=(eval('newcls._' + key) == attr.__class__.gid),
                                       enable_typechecks = False)
                    setattr(newcls, '__' + key, rel)

            else:
                ####  no default, nothing given
                LOG.warning('no sql column generated for attr %s, %r', key, attr)
        DeclarativeMetaType.__add_class_mapping_attributes(newcls, mapped_parent)
        return newcls


    @staticmethod
    def __add_class_mapping_attributes(newcls, mapped_parent):
        """
        Add Column ID and update __mapper_args__
        """
        #### Determine best FOREIGN KEY  
        mapped_parent = mapped_parent[0]
        fkparentid = mapped_parent.__tablename__ + '.id' 
        ### Update __mapper_args__ SQL_ALCHEMY attribute.    
        if newcls.__tablename__:
            LOG.debug('cls %r has dtparent %r', newcls, mapped_parent)
            LOG.debug('%r using %r as id foreignkey', newcls, fkparentid)
            column_id = sqlalchemy.Column('id', sqlalchemy.Integer,
                                          sqlalchemy.ForeignKey(fkparentid, ondelete="CASCADE"), primary_key=True)
            setattr(newcls, 'id', column_id)
            ### We can not use such a backref for cascading deletes, as we will have a cyclic dependency
            # (DataType > Mapped DT > Operation).
#            rel = relationship(mapped_parent, primaryjoin=(eval('newcls.id')==mapped_parent.id),
#                               backref = backref('__' +newcls.__name__, cascade="delete"))
#            setattr(newcls, '__id_' + mapped_parent.__name__, rel)
            mapper_arg = {}
            kwd = newcls.trait.inits.kwd
            if hasattr(newcls, '__mapper_args__'):
                mapper_arg = getattr(newcls, '__mapper_args__')

            if 'polymorphic_on' in mapper_arg and isinstance(mapper_arg['polymorphic_on'], (str, unicode)):
                discriminator_name = mapper_arg['polymorphic_on']
                LOG.debug("Polymorphic_on %s - %s " % (newcls.__name__, discriminator_name))
                mapper_arg['polymorphic_on'] = getattr(newcls, '_' + discriminator_name)
            mapper_arg['inherit_condition'] = (newcls.id == mapped_parent.id)
            if 'exclude_properties' in mapper_arg:
                del mapper_arg['exclude_properties']
                del mapper_arg['inherits']
            setattr(newcls, '__mapper_args__', mapper_arg)

            
TypeBase = declarative_base(cls=Type, name='TypeBase', metaclass=DeclarativeMetaType)


