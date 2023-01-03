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
Here we define entities related to user and project.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""

from datetime import datetime
import uuid

from sqlalchemy import Boolean, Integer, String, DateTime, Column, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm.collections import attribute_mapped_collection
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core import utils
from tvb.core.entities.exportable import Exportable
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neotraits.db import Base

LOG = get_logger(__name__)

# Constants for User Roles.
ROLE_ADMINISTRATOR = "ADMINISTRATOR"
ROLE_CLINICIAN = "CLINICIAN"
ROLE_RESEARCHER = "RESEARCHER"

USER_ROLES = [ROLE_ADMINISTRATOR, ROLE_CLINICIAN, ROLE_RESEARCHER]


class User(Base):
    """
    Contains the users information.
    """
    __tablename__ = 'USERS'

    id = Column(Integer, primary_key=True)
    gid = Column(String, unique=True)
    username = Column(String, unique=True)
    display_name = Column(String)
    password = Column(String)
    email = Column(String, nullable=True)
    role = Column(String)
    validated = Column(Boolean)
    selected_project = Column(Integer)

    preferences = association_proxy('user_preferences', 'value',
                                    creator=lambda k, v: UserPreferences(key=k, value=v))

    def __init__(self, login, display_name, password, email=None, validated=True, role=ROLE_RESEARCHER, gid=None):
        self.username = login
        self.display_name = display_name
        self.password = password
        self.email = email
        self.validated = validated
        self.role = role
        if gid is None:
            gid = uuid.uuid4().hex
        self.gid = gid

    def __repr__(self):
        return "<USER('%s','%s','%s','%s','%s','%s', %s)>" % (self.username, self.display_name, self.password,
                                                              self.email, self.validated, self.role,
                                                              str(self.selected_project))

    def is_administrator(self):
        """Return a boolean, saying if current user has role Administrator"""
        return self.role == ROLE_ADMINISTRATOR

    def is_online_help_active(self):
        """
        This method returns True if this user should see online help.
        """
        is_help_active = True
        if UserPreferences.ONLINE_HELP_ACTIVE in self.preferences:
            flag_str = self.preferences[UserPreferences.ONLINE_HELP_ACTIVE]
            is_help_active = utils.string2bool(flag_str)

        return is_help_active

    def switch_online_help_state(self):
        """
        This method changes the state of the OnlineHelp Active flag.
        """
        self.preferences[UserPreferences.ONLINE_HELP_ACTIVE] = str(not self.is_online_help_active())

    def set_viewers_color_scheme(self, color_scheme):
        self.preferences[UserPreferences.VIEWERS_COLOR_SCHEME] = color_scheme

    def get_viewers_color_scheme(self):
        if UserPreferences.VIEWERS_COLOR_SCHEME not in self.preferences:
            self.preferences[UserPreferences.VIEWERS_COLOR_SCHEME] = "linear"

        return self.preferences[UserPreferences.VIEWERS_COLOR_SCHEME]

    def set_project_structure_grouping(self, first, second):
        k = UserPreferences.PROJECT_STRUCTURE_GROUPING
        self.preferences[k] = "%s,%s" % (first, second)

    def get_project_structure_grouping(self):
        k = UserPreferences.PROJECT_STRUCTURE_GROUPING
        if k not in self.preferences:
            self.preferences[k] = "%s,%s" % (DataTypeMetaData.KEY_STATE, DataTypeMetaData.KEY_SUBJECT)
        return self.preferences[k].split(',')

    def set_preference(self, key, token):
        self.preferences[key] = token

    def get_preference(self, key):
        if key in self.preferences:
            return self.preferences[key]
        if hasattr(self, key):
            return getattr(self, key)
        return ""



class UserPreferences(Base):
    """
    Contains the user preferences data.
    """
    __tablename__ = 'USER_PREFERENCES'

    ONLINE_HELP_ACTIVE = "online_help_active"
    VIEWERS_COLOR_SCHEME = "viewers_color_scheme"
    PROJECT_STRUCTURE_GROUPING = "project_structure_grouping"
    fk_user = Column(Integer, ForeignKey('USERS.id'), primary_key=True)
    key = Column(String, primary_key=True)
    value = Column(String)

    user = relationship(User, backref=backref("user_preferences", cascade="all, delete-orphan", lazy='joined',
                                              collection_class=attribute_mapped_collection("key")))

    def __repr__(self):
        return 'UserPreferences: %s - %s' % (self.key, self.value)


class Project(Base, Exportable):
    """
    Contains the Projects informations and who is the administrator.
    """
    __tablename__ = 'PROJECTS'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    description = Column(String)
    last_updated = Column(DateTime)
    fk_admin = Column(Integer, ForeignKey('USERS.id'))
    gid = Column(String, unique=True)
    version = Column(Integer)
    disable_imports = Column(Boolean, default=False)
    max_operation_size = Column(Integer)

    administrator = relationship(User)

    ### Transient Attributes
    # todo: remove these as they are used only to transfer info to the templates. Have incorrect values prior to the templating phase.
    # send the status to the templates differently
    operations_finished = 0
    operations_started = 0
    operations_error = 0
    operations_canceled = 0
    operations_pending = 0

    members = []

    def __init__(self, name, fk_admin, max_operation_size, description='', disable_imports=False):
        self.name = name
        self.fk_admin = fk_admin
        self.max_operation_size = max_operation_size
        self.description = description
        self.disable_imports = disable_imports
        self.gid = utils.generate_guid()
        self.version = TvbProfile.current.version.PROJECT_VERSION

    def refresh_update_date(self):
        """Mark entity as being changed NOW. (last_update field)"""
        self.last_updated = datetime.now()

    def __repr__(self):
        return "<Project('%s', '%s')>" % (self.name, self.fk_admin)

    def to_dict(self):
        """
        Overwrite superclass method to add required changes.
        """
        _, base_dict = super(Project, self).to_dict(excludes=['id', 'fk_admin', 'administrator', 'trait'])
        return self.__class__.__name__, base_dict

    def from_dict(self, dictionary, user_id):
        """
        Add specific attributes from a input dictionary.
        """
        self.name = dictionary['name']
        self.description = dictionary['description']
        self.last_updated = datetime.now()
        self.gid = dictionary['gid']
        self.fk_admin = user_id
        self.version = int(dictionary['version'])
        return self


class User_to_Project(Base):
    """
    Multiple Users can be members of a given Project.
    """
    __tablename__ = 'USERS_TO_PROJECTS'

    id = Column(Integer, primary_key=True)
    fk_user = Column(Integer, ForeignKey('USERS.id', ondelete="CASCADE"))
    fk_project = Column(Integer, ForeignKey('PROJECTS.id', ondelete="CASCADE"))

    def __init__(self, user, case):

        if type(user) == int:
            self.fk_user = user
        else:
            self.fk_user = user.id

        if type(case) == int:
            self.fk_project = case
        else:
            self.fk_project = case.id
