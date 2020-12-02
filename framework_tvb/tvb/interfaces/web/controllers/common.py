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
Constants and functions used by all controllers
Custom exceptions

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from copy import copy
import cherrypy
from tvb.basic.exceptions import TVBException

# These are global constants, used for session attributes and template variables.
# Message, Current Project and User values are stored in session, because they
# need to be translated between multiple pages.
# The rest of the values are stored in the template dictionary.

TYPE_ERROR = "ERROR"
TYPE_WARNING = "WARNING"
TYPE_INFO = "INFO"
TYPE_IMPORTANT = "IMPORTANT"

KEY_KEYCLOAK_WEB = "keycloakConfigWebFile"
KEY_CURRENT_VERSION = "currentVersion"
KEY_CURRENT_JS_VERSION = "currentVersionJS"
KEY_SESSION = "session"
KEY_SESSION_TREE = "treeSessionKey"
KEY_USER = "user"
KEY_SHOW_ONLINE_HELP = "showOnlineHelp"
KEY_MESSAGE = "message"
KEY_MESSAGE_TYPE = "messageType"
KEY_PROJECT = "selectedProject"
KEY_ERRORS = "errors"
KEY_FORM_DATA = "data"
KEY_PARAMETERS_CONFIG = "param_checkbox_config"
KEY_FIRST_RUN = "first_run"
KEY_LINK_ANALYZE = "analyzeCategoryLink"
KEY_LINK_CONNECTIVITY_TAB = "connectivityTabLink"
KEY_TITLE = "title"
KEY_ADAPTER = "currentAlgoId"
KEY_SECTION = "section_name"
KEY_SUB_SECTION = 'subsection_name'
KEY_INCLUDE_RESOURCES = 'includedResources'
KEY_SUBMENU_LIST = 'submenu_list'
KEY_SUBMIT_LINK = 'submitLink'
KEY_DISPLAY_MENU = "displayControl"
KEY_PARENT_DIV = "parent_div"
# User section and settings section specific
KEY_IS_RESTART = "tvbRestarted"
KEY_INCLUDE_TOOLTIP = "includeTooltip"
KEY_WRAP_CONTENT_IN_MAIN_DIV = "wrapContentInMainDiv"
KEY_CURRENT_TAB = "currentTab"

KEY_BURST_CONFIG = 'burst_configuration'
KEY_SIMULATOR_CONFIG = 'simulator_configuration'
KEY_IS_SIMULATOR_COPY = 'is_simulator_copy'
KEY_IS_SIMULATOR_LOAD = 'is_simulator_load'
KEY_LAST_LOADED_FORM_URL = 'last_loaded_form_url'
KEY_BACK_PAGE = "back_page_link"
KEY_SECTION_TITLES = "section_titles"
KEY_SUBSECTION_TITLES = "sub_section_titles"
KEY_IS_SIMULATOR_BRANCH = "is_branch"

# Overlay specific keys
KEY_OVERLAY_TITLE = "overlay_title"
KEY_OVERLAY_DESCRIPTION = "overlay_description"
KEY_OVERLAY_CLASS = "overlay_class"
KEY_OVERLAY_CONTENT_TEMPLATE = "overlay_content_template"
KEY_OVERLAY_TABS_HORIZONTAL = "overlay_tabs_horizontal"
KEY_OVERLAY_TABS_VERTICAL = "overlay_tabs_vertical"
KEY_OVERLAY_INDEXES = "overlay_indexes"
KEY_OVERLAY_PAGINATION = "show_overlay_pagination"
KEY_OVERLAY_PREVIOUS = "action_overlay_previous"
KEY_OVERLAY_NEXT = "action_overlay_next"


def set_message(msg, m_type=TYPE_INFO):
    """ Set in session a message of a given type"""
    cherrypy.session.acquire_lock()
    cherrypy.session[KEY_MESSAGE] = msg
    cherrypy.session[KEY_MESSAGE_TYPE] = m_type
    cherrypy.session.release_lock()


def pop_message_from_session():
    """
    Pops the message stored in the session and it's type
    If no message is present returns an empty info message
    :returns: a message dict
    """
    m_type = remove_from_session(KEY_MESSAGE_TYPE)
    msg = remove_from_session(KEY_MESSAGE)

    if m_type is None:
        m_type = TYPE_INFO

    if msg is None:
        msg = ""

    return {KEY_MESSAGE: msg, KEY_MESSAGE_TYPE: m_type}


def get_message_from_session():
    m_type = get_from_session(KEY_MESSAGE_TYPE)
    msg = get_from_session(KEY_MESSAGE)
    return msg, m_type


def set_error_message(msg):
    """ Set error message in session"""
    set_message(msg, TYPE_ERROR)


def set_warning_message(msg):
    """ Set warning message in session"""
    set_message(msg, TYPE_WARNING)


def set_info_message(msg):
    """ Set info message in session"""
    set_message(msg, TYPE_INFO)


def set_important_message(msg):
    """ Set info message in session"""
    set_message(msg, TYPE_IMPORTANT)


def get_from_session(attribute):
    """ check if something exists in session and return"""
    return cherrypy.session.get(attribute)


def has_error_message():
    """ check if the session contains an error message """
    msg, msg_type = get_message_from_session()
    return msg_type == TYPE_ERROR


def remove_from_session(key):
    """ Remove from session an attributes if exists."""
    cherrypy.session.acquire_lock()
    if key in cherrypy.session:
        result = copy(cherrypy.session[key])
        del cherrypy.session[key]
        cherrypy.session.release_lock()
        return result
    cherrypy.session.release_lock()
    return None


def expire_session():
    """
    Expires and cleans current session.
    """
    # remove all session items
    cherrypy.session.acquire_lock()
    cherrypy.session.clear()
    # clear any caches held by cherrypy
    cherrypy.session.clean_up()
    # expire client side cookie
    cherrypy.lib.sessions.expire()
    cherrypy.session.release_lock()


def add2session(key, value):
    """ Set in session, at a key, a value"""
    cherrypy.session.acquire_lock()
    cherrypy.session[key] = value
    cherrypy.session.release_lock()


def get_logged_user():
    """Get current logged User from session"""
    return get_from_session(KEY_USER)


def get_current_project():
    """Get current Project from session"""
    return get_from_session(KEY_PROJECT)


def clean_project_data_from_session():
    remove_from_session(KEY_PROJECT)
    remove_from_session(KEY_SIMULATOR_CONFIG)
    remove_from_session(KEY_LAST_LOADED_FORM_URL)
    remove_from_session(KEY_BURST_CONFIG)
    remove_from_session(KEY_IS_SIMULATOR_BRANCH)
    add2session(KEY_IS_SIMULATOR_LOAD, False)


class NotAllowed(TVBException):
    """
    Raised when accessing a resource is not allowed
    """

    def __init__(self, message, redirect_url):
        TVBException.__init__(self, message)
        self.redirect_url = redirect_url
        self.status = 403


class NotAuthenticated(NotAllowed):
    """
    Raised when accessing a protected method with no user logged in
    """

    def __init__(self, message, redirect_url):
        NotAllowed.__init__(self, message, redirect_url)
        self.status = 401


class MissingDataException(TVBException):

    def __init__(self, message):
        TVBException.__init__(self, message)


class InvalidFormValues(TVBException):
    """
    Exception to be thrown in case of existing
    some invalid values in a form.
    """

    def __init__(self, message, error_dict=None):
        TVBException.__init__(self, message)
        self.error_dict = error_dict

    def display_full_errors(self):
        if self.error_dict:
            result = {}
            for name, item in self.error_dict.items():
                result[name] = str(item)
            return self.message, str(result).replace(',', ',\n')
        return self.message, ""
