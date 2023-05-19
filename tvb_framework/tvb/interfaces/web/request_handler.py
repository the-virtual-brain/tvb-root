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
.. moduleauthor:: calin.pavel <calin.pavel@codemart.ro>
"""

import os
import six
import shutil
import cherrypy
import tvb.interfaces.web.controllers.base_controller as bc
from tvb.basic.logger.builder import get_logger

# Constants for upload
from tvb.basic.profile import TvbProfile
from tvb.interfaces.web.controllers.common import get_from_session

CONTENT_LENGTH_KEY = 'content-length'

# Module logger
LOG = get_logger(__name__)


class RequestHandler(object):
    """
    This class contains different methods that can be used to enhance
    request processing. They are called at different moments depending the
    way they were registered in cherrypy configuration.
    """


    @staticmethod
    def check_upload_size():
        """
        This method checks if the uploaded content exceeds a given size
        """
        # convert the header keys to lower case
        lcHDRS = {}
        for key, val in six.iteritems(cherrypy.request.headers):
            lcHDRS[key.lower()] = val

        if CONTENT_LENGTH_KEY in lcHDRS:
            size = float(lcHDRS[CONTENT_LENGTH_KEY])
            if size > cherrypy.server.max_request_body_size:
                raise cherrypy.HTTPRedirect(TvbProfile.current.web.DEPLOY_CONTEXT + "/tvb?error=True")


    @staticmethod
    def clean_files_on_disk():
        """
        This method is executed at the end of a request and checks if there is any
        file which should be deleted on disk.
        """
        files_list = get_from_session(bc.FILES_TO_DELETE_ATTR)
        if files_list is not None and len(files_list) > 0:
            for (i, file_to_delete) in enumerate(files_list):
                if os.path.exists(file_to_delete):
                    try:
                        LOG.debug("End of request - deleting file/folder:%s" % file_to_delete)
                        if os.path.isfile(file_to_delete):
                            os.remove(file_to_delete)
                        else:
                            shutil.rmtree(file_to_delete)

                        # Delete success - now remove file from list
                        del files_list[i]
                    except Exception as exc:
                        LOG.error("Could not delete file/folder: %s" % file_to_delete)
                        LOG.exception(exc)
                else:
                    # File not found on disk, so we remove it from list
                    del files_list[i]
