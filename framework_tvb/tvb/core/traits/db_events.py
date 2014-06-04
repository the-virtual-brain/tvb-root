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
Python Triggers on DB operations.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy import event
from sqlalchemy.orm import mapper
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.types_basic import MapAsJson
from tvb.basic.traits.types_mapped import MappedType


# Logging support
LOG = get_logger(__name__)
# Refer SQLalchemy specific events
EVENT_LOAD = 'load'
EVENT_BEFORE_INSERT = 'before_insert'


def initialize_on_load(target, _):
    """
    Call this when any entity is loaded from DB.
    In case the entity inherits from MappedType, load associated relationships.
    e.g. TimeSeriesRegion.regions(of type Connectivity) should be populated with:
    
        - Connectivity instance, and not be a GID, as is default after DB storage.
    """
    if MappedType not in target.__class__.mro():
        return
    all_class_traits = getattr(target, 'trait', {})
    target.trait = all_class_traits.copy()
    LOG.debug("Custom Load event called for class:" + str(target.__class__.__name__))
    
    for key, attr in all_class_traits.iteritems():
        kwd = attr.trait.inits.kwd
        if (kwd.get('db', True) and (MappedType in attr.__class__.mro()) 
            and hasattr(target, '__' + key) and getattr(target, '__' + key) is not None):
            ### This attribute has a relationship associated
            loaded_entity = getattr(target, '__' + key)
            loaded_entity.trait.value = loaded_entity.initialize()
            loaded_entity.trait.name = key
            loaded_entity.trait.bound = True
            target.trait[key] = loaded_entity
            
    target.initialize()
    
    
    
def fill_before_insert(_, _ignored, target):
    """
    Trigger before storing MappedEntities in DB.
    Any time, when attaching an entity to the session, make sure 
    meta-data are cascaded on update-able attributes automatically.
    """
    if MappedType not in target.__class__.mro():
        return

    # Call apply_compute
    target.configure()
        
    # Fix object references
    all_class_traits = getattr(target, 'trait', {})
    for key, attr in all_class_traits.iteritems():
        kwd = attr.trait.inits.kwd
        # Here we try to resolve references in case _key is set 
        # and not key and __key
        if (kwd.get('db', True) and isinstance(attr, MappedType)
            and (hasattr(target, '_' + key) 
                 and getattr(target, '_' + key) is not None)
            and (hasattr(target, '__' + key) 
                 and getattr(target, '__' + key) is None)
                ):
            ref_entity = getattr(target, '_' + key)
            setattr(target, key, ref_entity)

    # Validate if the object can be stored
    target.validate()

    # Fix String attributes                
    # This HAVE to be executed LAST because any get() for Json attribute will 
    # restore it's value                    
    for key, attr in all_class_traits.iteritems():
        kwd = attr.trait.inits.kwd
        if (kwd.get('db', True) and isinstance(attr, MapAsJson)
                and hasattr(target, '_' + key)):
            field_value = getattr(target, '_' + key)
            if field_value is not None:
                setattr(target, '_' + key, attr.to_json(field_value))
                                


 
def attach_db_events():   
    """
    Attach events to all mapped tables.
    """
    event.listen(mapper, EVENT_LOAD, initialize_on_load)
    event.listen(mapper, EVENT_BEFORE_INSERT, fill_before_insert)






