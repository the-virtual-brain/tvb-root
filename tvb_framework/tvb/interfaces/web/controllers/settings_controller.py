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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import subprocess
import threading
import cherrypy
import formencode
from time import sleep
from formencode import validators
from tvb.basic.profile import TvbProfile
from tvb.core.services.exceptions import InvalidSettingsException
from tvb.core.services.settings_service import SettingsService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.decorators import check_admin, using_template, jsonify, handle_error
from tvb.interfaces.web.controllers.users_controller import UserController


@traced
class SettingsController(UserController):
    """
    Controller for TVB-Settings web page.
    Inherit from UserController, to have the same fill_default_attributes method (with versionInfo).
    """

    def __init__(self):
        UserController.__init__(self)
        self.settingsservice = SettingsService()

    @cherrypy.expose
    @handle_error(redirect=True)
    @using_template('user/base_user')
    @check_admin
    def settings(self, save_settings=False, **data):
        """Main settings page submit and get"""
        template_specification = dict(mainContent="settings/system_settings", title="System Settings")
        if save_settings:
            try:
                form = SettingsForm()
                data = form.to_python(data)
                isrestart, isreset = self.settingsservice.save_settings(**data)
                if isrestart:
                    thread = threading.Thread(target=self._restart_services, kwargs={'should_reset': isreset})
                    thread.start()
                    common.add2session(common.KEY_IS_RESTART, True)
                    common.set_important_message('Please wait until TVB is restarted properly!')
                    self.redirect('/tvb')
                # Here we will leave the same settings page to be displayed.
                # It will continue reloading when CherryPy restarts.
            except formencode.Invalid as excep:
                template_specification[common.KEY_ERRORS] = excep.unpack_errors()
            except InvalidSettingsException as excep:
                self.logger.error('Invalid settings!  Exception %s was raised' % (str(excep)))
                common.set_error_message(excep.message)
        template_specification.update({'keys_order': self.settingsservice.KEYS_DISPLAY_ORDER,
                                       'config_data': self.settingsservice.configurable_keys,
                                       common.KEY_FIRST_RUN: TvbProfile.is_first_run()})
        return self.fill_default_attributes(template_specification)

    def _restart_services(self, should_reset):
        """
        Restart CherryPy Backend.
        """
        sleep(1)
        cherrypy.engine.exit()

        self.logger.info("Waiting for CherryPy to shut down ... ")

        sleep(5)

        python_path = TvbProfile.current.PYTHON_INTERPRETER_PATH
        try:
            import tvb_bin
            proc_params = [python_path, '-m', 'tvb_bin.app', 'start', TvbProfile.CURRENT_PROFILE_NAME]
            if should_reset:
                proc_params.append('-reset')
            subprocess.Popen(proc_params, shell=False)
        except ImportError:
            proc_params = [python_path, '-m', 'tvb.interfaces.web.run', TvbProfile.CURRENT_PROFILE_NAME, "tvb.config"]
            if should_reset:
                proc_params.append('reset')
            subprocess.Popen(proc_params, shell=False).communicate()

        self.logger.info("Starting CherryPy again ... ")

    @cherrypy.expose
    @handle_error(redirect=False)
    @jsonify
    def check_db_url(self, **data):
        """
        Action on DB-URL validate button.
        """
        try:
            storage_path = data[self.settingsservice.KEY_STORAGE]
            if os.path.isfile(storage_path):
                raise InvalidSettingsException('TVB Storage should be set to a folder and not a file.')
            if not os.path.isdir(storage_path):
                try:
                    os.mkdir(storage_path)
                except OSError:
                    return {'status': 'not ok',
                            'message': 'Could not create root storage for TVB. Please check write permissions!'}
            self.settingsservice.check_db_url(data[self.settingsservice.KEY_DB_URL])
            return {'status': 'ok', 'message': 'The database URL is valid.'}
        except InvalidSettingsException as excep:
            self.logger.error(excep)
            return {'status': 'not ok', 'message': 'The database URL is not valid.'}


class DiskSpaceValidator(formencode.FancyValidator):
    """
    Custom validator for TVB disk space / user.
    """

    def _convert_to_python(self, value, _):
        """
        Validation required method.
        :param value is user-specified value, in MB
        """
        try:
            value = int(value)
            return value
        except ValueError:
            raise formencode.Invalid('Invalid disk space %s. Should be number' % value, value, None)


class PortValidator(formencode.FancyValidator):
    """
    Custom validator for OS Port number.
    """

    def _convert_to_python(self, value, _):
        """ 
        Validation required method.
        """
        try:
            value = int(value)
        except ValueError:
            raise formencode.Invalid('Invalid port %s. Should be number between 0 and 65535.' % value, value, None)
        if 0 < value < 65535:
            return value
        else:
            raise formencode.Invalid('Invalid port number %s. Should be in interval [0, 65535]' % value, value, None)


class ThreadNrValidator(formencode.FancyValidator):
    """
    Custom validator number of threads.
    """

    def _convert_to_python(self, value, _):
        """ 
        Validation required method.
        """
        try:
            value = int(value)
        except ValueError:
            raise formencode.Invalid('Invalid number %s. Should be number between 1 and 16.' % value, value, None)
        if 0 < value < 17:
            return value
        else:
            raise formencode.Invalid('Invalid number %d. Should be in interval [1, 16]' % value, value, None)


class SurfaceVerticesNrValidator(formencode.FancyValidator):
    """
    Custom validator for the number of vertices allowed for a surface
    """
    # This limitation is given by our Max number of colors in pick mechanism
    MAX_VALUE = 256 * 256 * 256 + 1

    def _convert_to_python(self, value, _):
        """ 
        Validation required method.
        """
        msg = 'Invalid value: %s. Should be a number between 1 and %d.'
        try:
            value = int(value)
            if 0 < value < self.MAX_VALUE:
                return value
            else:
                raise formencode.Invalid(msg % (str(value), self.MAX_VALUE), value, None)
        except ValueError:
            raise formencode.Invalid(msg % (value, self.MAX_VALUE), value, None)


class AsciiValidator(formencode.FancyValidator):
    """
    Allow only ascii strings
    """

    def _convert_to_python(self, value, _):
        try:
            return str(value).encode('ascii')
        except UnicodeError:
            raise formencode.Invalid('Invalid ascii string %s' % value, '', None)


class SettingsForm(formencode.Schema):
    """
    Validate Settings Page inputs.
    """

    ADMINISTRATOR_NAME = formencode.All(validators.UnicodeString(not_empty=True), validators.PlainText())
    ADMINISTRATOR_DISPLAY_NAME = formencode.All(validators.UnicodeString(not_empty=True), validators.PlainText())
    ADMINISTRATOR_PASSWORD = validators.UnicodeString(not_empty=True)
    ADMINISTRATOR_EMAIL = validators.Email(not_empty=True)

    WEB_SERVER_PORT = PortValidator()

    SELECTED_DB = validators.UnicodeString(not_empty=True)
    URL_VALUE = validators.UnicodeString(not_empty=True)
    DEPLOY_CLUSTER = validators.Bool()
    CLUSTER_SCHEDULER = validators.UnicodeString(not_empty=True)

    KEYCLOAK_CONFIGURATION = validators.UnicodeString()
    KEYCLOAK_WEB_CONFIGURATION = validators.UnicodeString()
    ENABLE_KEYCLOAK_LOGIN = validators.Bool()
    TVB_STORAGE = validators.UnicodeString(not_empty=True)
    USR_DISK_SPACE = DiskSpaceValidator(not_empty=True)
    MAXIMUM_NR_OF_THREADS = ThreadNrValidator()
    MAXIMUM_NR_OF_VERTICES_ON_SURFACE = SurfaceVerticesNrValidator()
    MAXIMUM_NR_OF_OPS_IN_RANGE = validators.Int(min=5, max=5000, not_empty=True)
