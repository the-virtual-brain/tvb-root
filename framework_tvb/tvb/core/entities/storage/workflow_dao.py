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

"""
DAO layer for WorkFlow and Burst entities.
"""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import or_
from tvb.core.entities.model.model_burst import Dynamic
from tvb.core.entities.model.model_operation import Algorithm, AlgorithmCategory
from tvb.core.entities.model.model_workflow import Portlet
from tvb.core.entities.storage.root_dao import RootDAO


class WorkflowDAO(RootDAO):
    """
    DAO layer for WorkFlow and Burst entities.
    """


    def get_non_validated_entities(self, reference_time):
        """
        Get a list of all categories, portlets and algorithm groups that were not found valid since the reference_time.
        Used in initializer on each start to filter out any entities that for some reason became invalid.
        :return tuple (list of entities to get invalidated) (list of entities to be removed)
        """
        try:
            stored_adapters = self.session.query(Algorithm
                                        ).filter(or_(Algorithm.last_introspection_check == None,
                                                     Algorithm.last_introspection_check < reference_time)).all()
            categories = self.session.query(AlgorithmCategory
                                        ).filter(AlgorithmCategory.last_introspection_check<reference_time).all()
            portlets = self.session.query(Portlet
                                        ).filter(Portlet.last_introspection_check < reference_time).all()
            result = stored_adapters + categories, portlets
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
            result = [], []
        return result

    def get_available_portlets(self, ):
        """
        Get all the stored portlets form the db.
        """
        portlets = []
        try:
            portlets = self.session.query(Portlet).order_by(Portlet.name).all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return portlets


    def get_portlet_by_identifier(self, portlet_identifier):
        """
        Given an identifer retieve the portlet that corresponds to it.
        """
        portlet = None
        try:
            portlet = self.session.query(Portlet).filter_by(algorithm_identifier=portlet_identifier).one()
        except NoResultFound:
            self.logger.debug("No portlet found with id=%s." % portlet_identifier)
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return portlet


    def get_portlet_by_id(self, portlet_id):
        """
        Given an portlet id retieve the portlet entity.
        """
        portlet = None
        try:
            portlet = self.session.query(Portlet).filter_by(id=portlet_id).one()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)

        return portlet

    def get_dynamics_for_user(self, user_id):
        try:
            return self.session.query(Dynamic).filter(Dynamic.fk_user == user_id).all()
        except SQLAlchemyError as exc:
            self.logger.exception(exc)
            return []


    def get_dynamic(self, dyn_id):
        try:
            return self.session.query(Dynamic).filter(Dynamic.id == dyn_id).one()
        except SQLAlchemyError as exc:
            self.logger.exception(exc)

    def get_dynamic_by_name(self, name):
        try:
            return self.session.query(Dynamic).filter(Dynamic.name == name).all()
        except SQLAlchemyError as exc:
            self.logger.exception(exc)
