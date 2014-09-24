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
DAO operation related to Users and Projects are defined here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from sqlalchemy import or_, and_, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.expression import desc
from sqlalchemy.orm.exc import NoResultFound
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.core.entities import model
from tvb.core.entities.storage.root_dao import RootDAO



class CaseDAO(RootDAO):
    """
    USER and PROJECT RELATED OPERATIONS
    """


    def get_user_by_id(self, user_id):
        """Retrieve USER entity by name."""
        user = None
        try:
            user = self.session.query(model.User).filter_by(id=user_id).one()
        except SQLAlchemyError:
            pass
        return user


    def get_user_by_name(self, name):
        """Retrieve USER entity by name."""
        user = None
        try:
            user = self.session.query(model.User).filter_by(username=name).one()
        except SQLAlchemyError:
            pass
        return user


    def get_system_user(self):
        """Retrieve System user by name."""
        user = None
        try:
            user = self.session.query(model.User).filter_by(username=cfg.SYSTEM_USER_NAME).one()
        except SQLAlchemyError:
            pass
        return user


    def count_users_for_name(self, name):
        """Retrieve the number of users in DB for a given name."""
        result = self.session.query(model.User).filter_by(username=name).count()
        return result


    def get_administrators(self):
        """Retrieve System user by name."""
        admins = None
        try:
            admins = self.session.query(model.User).filter_by(role=model.ROLE_ADMINISTRATOR).all()
        except SQLAlchemyError:
            pass
        return admins


    def get_all_users(self, different_name=' ', page_start=0, page_end=20, is_count=False):
        """Retrieve all USERS in DB, except current user and system user."""
        try:
            query = self.session.query(model.User
                                       ).filter(model.User.username != different_name
                                                ).filter(model.User.username != cfg.SYSTEM_USER_NAME)
            if is_count:
                result = query.count()
            else:
                result = query.order_by(model.User.username).offset(max(page_start, 0)).limit(max(page_end, 0)).all()
            return result
        except NoResultFound:
            self.logger.warning("No users found. Maybe database is empty.")
            raise


    def get_user_by_name_email(self, email, name_hint=""):
        """
        Find a user by email address and name.

        :param email: Valid email address string, to search for its exact match in DB
        :param name_hint: string for a user's name; to search with like in DB

        :return: None if none or more than one users matches the criteria.
        """
        user = None

        if name_hint:
            try:
                ### In case of multiple users: first try to find exact match for the given username
                user = self.session.query(model.User).filter_by(email=email).filter_by(username=name_hint).one()
            except SQLAlchemyError:
                ### Ignore
                pass

        if user is None:
            try:
                user = self.session.query(model.User).filter_by(email=email
                                ).filter(model.User.username.ilike('%' + name_hint + '%')).one()
            except SQLAlchemyError:
                self.logger.exception("Could not get a single user by email " + email + " and name " + name_hint)

        return user


    def delete_user(self, user_id):
        """Remove USER entity by ID."""
        user = self.session.query(model.User).filter_by(id=user_id).one()
        self.session.delete(user)
        self.session.commit()


    def get_user_for_datatype(self, dt_id):
        """Get the user for the datatype id"""
        try:
            datatype = self.session.query(model.DataType).filter_by(id=dt_id).one()
            datatype.parent_operation
            user = datatype.parent_operation.user
        except SQLAlchemyError, ex:
            self.logger.exception(ex)
            user = None
        return user


    #
    # PROJECT RELATED OPERATIONS
    #

    def get_project_by_id(self, project_id):
        """Retrieve PROJECT entity for a given identifier.
           THROW SqlException when not found."""
        prj = self.session.query(model.Project).filter_by(id=project_id).one()
        prj.administrator
        return prj


    def get_project_by_gid(self, project_gid):
        """Retrieve PROJECT entity for a given identifier.
           THROW SqlException when not found."""
        prj = self.session.query(model.Project).filter_by(gid=project_gid).one()
        prj.administrator
        return prj


    def delete_project(self, project_id):
        """Remove PROJECT entity by ID."""
        project = self.session.query(model.Project).filter_by(id=project_id).one()
        self.session.delete(project)
        linked_users = self.session.query(model.User
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
            total_size = self.session.query(func.sum(model.DataType.disk_size)).join(model.Operation
                                        ).filter(model.Operation.fk_launched_in == project_id).scalar()
            return total_size or 0
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return -1


    def count_projects_for_name(self, name, different_id):
        """Retrieve the number of projects with a given name currently in DB."""
        if different_id is not None:
            number = self.session.query(model.Project
                                        ).filter_by(name=name).filter(model.Project.id != different_id).count()
        else:
            number = self.session.query(model.Project).filter_by(name=name).count()
        return number


    def get_all_projects(self, page_start=0, page_end=20, is_count=False):
        """
        Retrieve all Project entities currently in the system.
        WARNING: use this wisely, as it might easily overflow the system.
        """
        query = self.session.query(model.Project)
        if is_count:
            result = query.count()
        else:
            result = query.offset(max(page_start, 0)).limit(max(page_end, 0)).all()
        return result


    def get_projects_for_user(self, user_id, page_start=0, page_end=20, is_count=False):
        """
        Return all projects a given user can access (administrator or not).
        """
        # First load projects that current user is administrator for.
        query = self.session.query(model.Project).join((model.User, model.Project.fk_admin == model.User.id)
                                ).outerjoin((model.User_to_Project,
                                             and_(model.Project.id == model.User_to_Project.fk_project,
                                                  model.User_to_Project.fk_user == user_id))
                                ).filter(or_(model.User.id == user_id, model.User_to_Project.fk_user == user_id)
                                ).order_by(desc(model.Project.id))
        if is_count:
            result = query.count()
        else:
            result = query.offset(max(page_start, 0)).limit(max(page_end, 0)).all()
            [project.administrator.username for project in result]
        return result


    def get_project_for_operation(self, operation_id):
        """
        Find parent project for current operation.
        """
        try:
            result = self.session.query(model.Project
                                        ).filter(model.Operation.fk_launched_in == model.Project.id
                                                 ).filter(model.Operation.id == operation_id).one()
            return result
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None


    def get_links_for_project(self, project_id):
        """
        :return all links refering to a given project_id
        """
        result = self.session.query(model.Links).filter(model.Links.fk_to_project == project_id).all()
        return result


    def get_link(self, dt_id, project_id):
        """
        :return link between a given DT and a given project id
        """
        try:
            result = self.session.query(model.Links).filter(model.Links.fk_from_datatype == dt_id
                                                        ).filter(model.Links.fk_to_project == project_id).one()
            return result
        except SQLAlchemyError, excep:
            self.logger.exception(excep)
            return None



    def get_linkable_projects_for_user(self, user_id, data_id):
        """
        Return all projects a given user can link some data given by a data_id to.
        """
        try:
            # First load projects that current user is administrator for.
            result = self.session.query(model.Project).join(model.User
                                        ).filter(model.User.id == user_id).order_by(model.Project.id).all()
            result.extend(self.session.query(model.Project).join(model.User_to_Project
                                             ).filter(model.User_to_Project.fk_user == user_id).all())
            linked_project_ids = self.session.query(model.Links.fk_to_project
                                                    ).filter(model.Links.fk_from_datatype == data_id).all()
            linked_project_ids = [i[0] for i in linked_project_ids]
            datatype = self.get_datatype_by_id(data_id)
            current_prj = self.session.query(model.Operation.fk_launched_in
                                             ).filter(model.Operation.id == datatype.fk_from_operation).one()
            if linked_project_ids:
                linked_project_ids.append(current_prj[0])
            else:
                linked_project_ids = [current_prj[0]]
            filtered_result = [entry for entry in result if entry.id not in linked_project_ids]
            [project.administrator.username for project in filtered_result]
            linked_project_ids.remove(current_prj[0])
            linked_projects = [entry for entry in result if entry.id in linked_project_ids]
            return filtered_result, linked_projects
        except Exception, excep:
            self.logger.exception(excep)
            return None, None


    def delete_members_for_project(self, project_id, members):
        """Remove all linked user to current project."""
        members = self.session.query(model.User_to_Project
                                     ).filter(model.User_to_Project.fk_project == project_id
                                              ).filter(model.User_to_Project.fk_user.in_(members)).all()
        [self.session.delete(m) for m in members]
        self.session.commit()


    def get_members_of_project(self, proj_id):
        """Retrieve USER entities with rights on current project."""
        users_members = self.session.query(model.User).join(model.User_to_Project
                                           ).filter(model.User_to_Project.fk_project == proj_id).all()
        return users_members


    def get_operation_numbers(self, proj_id):
        """
        Count total number of operations started for current project.
        """
        fns = self.session.query(model.Operation).filter_by(fk_launched_in=proj_id
                                 ).filter_by(status=model.STATUS_FINISHED).count()
        sta = self.session.query(model.Operation).filter_by(fk_launched_in=proj_id
                                 ).filter_by(status=model.STATUS_STARTED).count()
        err = self.session.query(model.Operation).filter_by(fk_launched_in=proj_id
                                 ).filter_by(status=model.STATUS_ERROR).count()
        canceled = self.session.query(model.Operation).filter_by(fk_launched_in=proj_id
                                      ).filter_by(status=model.STATUS_CANCELED).count()
        return fns, sta, err, canceled
