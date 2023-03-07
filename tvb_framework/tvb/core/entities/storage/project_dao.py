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
DAO operation related to Users and Projects are defined here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import or_, and_, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import desc
from tvb.basic.profile import TvbProfile
from tvb.core.entities.model.model_datatype import DataType, Links
from tvb.core.entities.model.model_operation import Operation
from tvb.core.entities.model.model_project import User, ROLE_ADMINISTRATOR, Project, User_to_Project
from tvb.core.entities.storage.root_dao import RootDAO, DEFAULT_PAGE_SIZE


class CaseDAO(RootDAO):
    """
    USER and PROJECT RELATED OPERATIONS
    """

    #
    # USER RELATED METHODS
    #

    def get_user_by_id(self, user_id):
        """Retrieve USER entity by ID."""
        user = None
        try:
            user = self.session.query(User).filter_by(id=user_id).one()
        except SQLAlchemyError:
            self.logger.exception("Could not retrieve user for id " + str(user_id))
        return user

    def get_user_by_name(self, name):
        """Retrieve USER entity by name."""
        user = None
        try:
            user = self.session.query(User).filter_by(username=name).one()
        except SQLAlchemyError:
            self.logger.debug("Could not retrieve user for name " + str(name))
        return user

    def get_user_by_gid(self, gid):
        """Retrieve USER entity by gid."""
        user = None
        try:
            user = self.session.query(User).filter_by(gid=gid).one()
        except SQLAlchemyError:
            self.logger.debug("Could not retrieve user for gid= " + gid)
        return user

    def get_system_user(self):
        """Retrieve System user from DB."""
        user = None
        sys_name = TvbProfile.current.web.admin.SYSTEM_USER_NAME
        try:
            user = self.session.query(User).filter_by(username=sys_name).one()
        except SQLAlchemyError:
            self.logger.exception("Could not retrieve system user " + str(sys_name))
        return user

    def count_users_for_name(self, name):
        """Retrieve the number of users in DB for a given name."""
        result = self.session.query(User).filter_by(username=name).count()
        return result

    def get_administrators(self):
        """Retrieve all users with Admin role"""
        admins = self.session.query(User).filter_by(role=ROLE_ADMINISTRATOR).all()
        return admins

    def get_all_users(self, different_names=None, page_start=0, page_size=DEFAULT_PAGE_SIZE, is_count=False):
        """Retrieve all USERS in DB, except given users and system user."""
        if different_names is None:
            different_names = []
        try:
            sys_name = TvbProfile.current.web.admin.SYSTEM_USER_NAME
            query = self.session.query(User
                                       ).filter(User.username.notin_(different_names)
                                                ).filter(User.username != sys_name)
            if is_count:
                result = query.count()
            else:
                result = query.order_by(User.username).offset(max(page_start, 0)).limit(max(page_size, 0)).all()
            return result
        except NoResultFound:
            self.logger.warning("No users found. Maybe database is empty.")
            raise

    def get_user_by_email(self, email, name_hint=""):
        """
        Find a user by email address and name.

        :param email: Valid email address string, to search for its exact match in DB
        :param name_hint: string for a user's name; to search with like in DB

        :return: None if none or more than one users matches the criteria.
        """
        user = None

        if name_hint:
            try:
                # In case of multiple users: first try to find exact match for the given username
                user = self.session.query(User).filter_by(email=email).filter_by(username=name_hint).one()
            except SQLAlchemyError:
                # Ignore
                pass

        if user is None:
            try:
                user = self.session.query(User).filter_by(email=email
                                                          ).filter(User.username.ilike('%' + name_hint + '%')).one()
            except SQLAlchemyError:
                self.logger.exception("Could not get a single user by email " + email + " and name " + name_hint)

        return user

    def get_user_for_datatype(self, dt_id):
        """Get the user who created a DT"""
        try:
            datatype = self.session.query(DataType).filter_by(id=dt_id).one()
            return datatype.parent_operation.user
        except SQLAlchemyError as ex:
            self.logger.exception(ex)
        return None

    def compute_user_generated_disk_size(self, user_id):
        """
        Do a SUM on DATA_TYPES table column DISK_SIZE, for the current user.
        :returns 0 when no DT are found, or SUM from DB.
        """
        try:
            total_size = self.session.query(func.sum(DataType.disk_size)).join(Operation
                                                                               ).filter(
                Operation.fk_launched_by == user_id).scalar()
            return total_size or 0
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return -1

    #
    # PROJECT RELATED METHODS
    #

    def get_project_by_id(self, project_id):
        """Retrieve PROJECT entity for a given identifier.
           THROW SqlException when not found."""
        prj = self.session.query(Project).filter_by(id=project_id).one()
        prj.administrator
        return prj

    def get_project_by_gid(self, project_gid):
        """Retrieve PROJECT entity for a given identifier.
           THROW SqlException when not found."""
        prj = self.session.query(Project).filter_by(gid=project_gid).one()
        prj.administrator
        return prj

    def get_project_by_name(self, project_name):
        """Retrieve PROJECT entity for a given name.
        THROW SQLException when not found."""
        prj = self.session.query(Project).filter_by(name=project_name).one()
        prj.administrator
        return prj

    def get_project_lazy_by_gid(self, project_gid):
        """Retrieve PROJECT entity for a given identifier.
           THROW SqlException when not found."""
        return self.session.query(Project).filter_by(gid=project_gid).one()

    def delete_project(self, project_id):
        """Remove PROJECT entity by ID."""
        project = self.session.query(Project).filter_by(id=project_id).one()
        self.session.delete(project)
        linked_users = self.session.query(User
                                          ).filter_by(selected_project=project_id).all()
        for user in linked_users:
            user.selected_project = None
        self.session.commit()

    def get_project_disk_size(self, project_id):
        """
        Do a SUM on DATA_TYPES table column DISK_SIZE, for the current project.
        :returns 0 when no DT are found, or SUM from DB.
        """
        try:
            total_size_for_dt = self.session.query(func.sum(DataType.disk_size)).join(Operation
                                                                               ).filter(
                Operation.fk_launched_in == project_id).scalar() or 0
            total_size = self.session.query(func.sum(Operation.view_model_disk_size)).filter(
                Operation.fk_launched_in == project_id).scalar() or 0
            total_size = total_size + total_size_for_dt
            return total_size
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return -1

    def count_projects_for_name(self, name, different_id):
        """Retrieve the number of projects with a given name currently in DB."""
        if different_id is not None:
            number = self.session.query(Project
                                        ).filter_by(name=name).filter(Project.id != different_id).count()
        else:
            number = self.session.query(Project).filter_by(name=name).count()
        return number

    def get_all_projects(self, page_start=0, page_size=DEFAULT_PAGE_SIZE, is_count=False):
        """
        Retrieve all Project entities currently in the system.
        WARNING: use this wisely, as it might easily overflow the system.
        """
        query = self.session.query(Project)
        if is_count:
            result = query.count()
        else:
            result = query.offset(max(page_start, 0)).limit(max(page_size, 0)).all()
        return result

    def get_projects_for_user(self, user_id, page_start=0, page_size=DEFAULT_PAGE_SIZE, is_count=False):
        """
        Return all projects a given user can access (administrator or not).
        """
        # First load projects that current user is administrator for.
        query = self.session.query(Project).join(User, Project.fk_admin == User.id
                                                 ).outerjoin(User_to_Project,
                                                              and_(Project.id == User_to_Project.fk_project,
                                                                   User_to_Project.fk_user == user_id)
                                                             ).filter(
            or_(User.id == user_id, User_to_Project.fk_user == user_id)
        ).order_by(desc(Project.id))
        if is_count:
            result = query.count()
        else:
            result = query.offset(max(page_start, 0)).limit(max(page_size, 0)).all()
            [project.administrator.username for project in result]
        return result

    def get_project_for_operation(self, operation_id):
        """
        Find parent project for current operation.
        THROW SqlException when not found.
        """
        result = self.session.query(Project
                                    ).filter(Operation.fk_launched_in == Project.id
                                             ).filter(Operation.id == operation_id).one()
        return result

    def get_links_to_project(self, project_id):
        """
        :return all links referring to a given project_id
        """
        result = self.session.query(Links).filter(Links.fk_to_project == project_id).all()
        return result

    def get_link(self, dt_id, project_id):
        """
        :return link between a given DT and a given project id
        """
        try:
            result = self.session.query(Links).filter(Links.fk_from_datatype == dt_id
                                                      ).filter(Links.fk_to_project == project_id).one()
            return result
        except SQLAlchemyError as excep:
            self.logger.exception(excep)
            return None

    def get_linkable_projects_for_user(self, user_id, data_id):
        """
        Return all projects a given user can link some data given by a data_id to.
        """
        try:
            # Load projects where the current user is Admin or Member
            result = self.session.query(Project).join(User
                                                      ).join(User_to_Project
                                                             ).filter(or_(User.id == user_id,
                                                                          User_to_Project.fk_user == user_id)
                                                                      ).all()
            linked_project_ids = self.session.query(Links.fk_to_project
                                                    ).filter(Links.fk_from_datatype == data_id).all()
            linked_project_ids = [i[0] for i in linked_project_ids]
            datatype = self.get_datatype_by_id(data_id)
            current_prj = self.session.query(Operation.fk_launched_in
                                             ).filter(Operation.id == datatype.fk_from_operation).one()
            if linked_project_ids:
                linked_project_ids.append(current_prj[0])
            else:
                linked_project_ids = [current_prj[0]]

            filtered_result = [entry for entry in result if entry.id not in linked_project_ids]
            [project.administrator.username for project in filtered_result]
            linked_project_ids.remove(current_prj[0])
            linked_projects = [entry for entry in result if entry.id in linked_project_ids]
            return filtered_result, linked_projects
        except Exception as excep:
            self.logger.exception(excep)
            return None, None

    def delete_members_for_project(self, project_id, members):
        """Remove all linked user to current project."""
        members = self.session.query(User_to_Project
                                     ).filter(User_to_Project.fk_project == project_id
                                              ).filter(User_to_Project.fk_user.in_(members)).all()
        [self.session.delete(m) for m in members]
        self.session.commit()

    def get_members_of_project(self, proj_id):
        """Retrieve USER entities with rights on current project."""
        users_members = self.session.query(User).join(User_to_Project
                                                      ).filter(User_to_Project.fk_project == proj_id).all()
        return users_members

    def add_members_to_project(self, proj_id, selected_user_ids):
        """Add link between Users and Project."""
        for u_id in selected_user_ids:
            self.store_entity(User_to_Project(u_id, proj_id))
