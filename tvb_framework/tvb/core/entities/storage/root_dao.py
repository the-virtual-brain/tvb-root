# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Base DAO behavior.

.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import importlib
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage.session_maker import SESSION_META_CLASS
from tvb.config import SIMULATION_DATATYPE_CLASS


DEFAULT_PAGE_SIZE = 200


class RootDAO(object, metaclass=SESSION_META_CLASS):
    """
    GLOBAL METHODS
    """
    session = None
    logger = get_logger(__name__)

    EXCEPTION_DATATYPE_GROUP = "DataTypeGroup"
    EXCEPTION_DATATYPE_SIMULATION = SIMULATION_DATATYPE_CLASS

    def store_entity(self, entity, merge=False):
        """
        Store in DB one generic entity.
        """
        self.logger.debug("We will store entity of type: %s with id %s" % (entity.__class__.__name__, str(entity.id)))

        if merge:
            self.session.merge(entity)
        else:
            self.session.add(entity)
        self.session.commit()

        self.logger.debug("After commit %s ID is %s" % (entity.__class__.__name__, str(entity.id)))

        saved_entity = self.session.query(entity.__class__).filter_by(id=entity.id).one()
        return saved_entity

    def store_entities(self, entities_list):
        """
        Store in DB a list of generic entities.
        """
        self.session.add_all(entities_list)
        self.session.commit()

        stored_entities = []
        for entity in entities_list:
            stored_entities.append(self.session.query(entity.__class__).filter_by(id=entity.id).one())
        return stored_entities

    def get_generic_entity(self, entity_type, filter_value, select_field="id"):
        """
        Retrieve an entity of entity_type, filtered by select_field = filter_value.
        """
        if isinstance(entity_type, str):
            classname = entity_type[entity_type.rfind(".") + 1:]
            module = importlib.import_module(entity_type[0: entity_type.rfind(".")])
            entity_class = getattr(module, classname)
            result = self.session.query(entity_class).filter(entity_class.__dict__[select_field] == filter_value).all()
        else:
            result = self.session.query(entity_type).filter(entity_type.__dict__[select_field] == filter_value).all()

        # Need this since entity has attributes loaded automatically on DB load from 
        # traited DB events. This causes the session to see the entity as dirty and issues
        # an invalid commit() which leaves the entity unattached to any sessions later on.
        self.session.expunge_all()
        return result

    def remove_entity(self, entity_class, entity_id):
        """ 
        Find entity by Id and Type, end then remove it.
        Return True, when entity was removed successfully, of False when exception.
        """
        try:
            entity = self.session.query(entity_class).filter_by(id=entity_id).one()
            self.session.delete(entity)
            self.session.commit()
            return True
        except NoResultFound:
            self.logger.info("Entity from class %s with id %s has been already removed." % (entity_class, entity_id))
            return True
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return False

    # DATA_TYPE BUT GENERIC METHODS

    def remove_datatype(self, gid):
        """
        When removing dataType, load fully so that sql-alchemy removes from all tables referenced.
        """
        data = self.session.query(DataType).filter(DataType.gid == gid).all()
        for entity in data:
            extended_ent = self.get_generic_entity(entity.module + "." + entity.type, entity.id)
            self.session.delete(extended_ent[0])
        self.session.commit()

    def get_datatype_by_id(self, data_id):
        """
        Retrieve DataType entity by ID.
        """
        result = self.session.query(DataType).filter_by(id=data_id).one()
        result.parent_operation.project
        return result

    def get_time_series_by_gid(self, data_gid):
        result = self.session.query(DataType).filter_by(gid=data_gid).one()
        result.data
        return result