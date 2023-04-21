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

from sqlalchemy import desc, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import aliased
from sqlalchemy.orm.exc import NoResultFound
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.storage.root_dao import RootDAO, DEFAULT_PAGE_SIZE


class BurstDAO(RootDAO):
    """
    DAO layer for Burst entities.
    """

    def get_bursts_for_project(self, project_id, page_start=0, page_size=DEFAULT_PAGE_SIZE, count=False):
        """Get latest 50 BurstConfiguration entities for the current project"""
        try:
            bursts = self.session.query(BurstConfiguration
                                        ).filter_by(fk_project=project_id
                                                    ).order_by(desc(BurstConfiguration.start_time))
            if count:
                return bursts.count()
            if page_size is not None:
                bursts = bursts.offset(max(page_start, 0)).limit(page_size)

            bursts = bursts.all()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            bursts = None
        return bursts

    def get_max_burst_id(self):
        """
        Return the maximum of the currently stored burst IDs to be used as the new burst name.
        This is not a thread-safe value, but we use it just for a label.
        """
        try:
            max_id = self.session.query(func.max(BurstConfiguration.id)).one()
            if max_id[0] is None:
                return 0
            return max_id[0]
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return 0

    def count_bursts_with_name(self, burst_name, project_id):
        """
        Return the number of burst already named 'custom_b%' and NOT 'custom_b%_%' in current project.
        """
        count = 0
        try:
            count = self.session.query(BurstConfiguration
                                    ).filter_by(fk_project=project_id
                                    ).filter(BurstConfiguration.name.like(burst_name + '_branch%')
                                    ).filter(BurstConfiguration.name.notlike(burst_name + '_branch%_branch%', escape='/')
                ).count()
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return count

    def get_burst_by_id(self, burst_id):
        """Get the BurstConfiguration entity with the given id"""
        try:
            burst = self.session.query(BurstConfiguration).filter_by(id=burst_id).one()
            burst.project
            return burst
        except NoResultFound:
            self.logger.debug("No burst found with id = {}".format(burst_id))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return None

    def get_burst_for_operation_id(self, operation_id, is_group=False):
        burst = None
        try:
            burst = self.get_burst_for_direct_operation_id(operation_id, is_group)
            if not burst:
                burst_alias = aliased(BurstConfiguration, flat=True)
                burst = self.session.query(burst_alias
                                           ).join(DataType, DataType.fk_parent_burst == burst_alias.gid
                                                  ).filter(DataType.fk_from_operation == operation_id).first()
        except NoResultFound:
            self.logger.debug("No burst found for operation id = %s" % (operation_id,))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return burst

    def get_burst_for_direct_operation_id(self, operation_id, is_group=False):
        burst = None
        try:
            if is_group:
                burst = self.session.query(BurstConfiguration
                                           ).filter(BurstConfiguration.fk_operation_group == operation_id).first()
            else:
                burst = self.session.query(BurstConfiguration
                                           ).filter(BurstConfiguration.fk_simulation == operation_id).first()
        except NoResultFound:
            self.logger.debug("No direct burst found for operation id = %s" % (operation_id,))
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
        return burst
