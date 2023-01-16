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
from tvb.basic.config.environment import Environment


def user_environment_execution(func):
    """
    Decorator that makes sure a function is executed in a 'user' environment,
    removing any TVB specific configurations that alter either LD_LIBRARY_PATH
    or LD_RUN_PATH.
    """

    def _remove_from_path(env_name, count, segment_marker):
        """
        For 'env_name' environment variable representing a path (e.g. LB_RUN_PATH), remove first 'count' segments.
        Remove only those path segments within the limit of 'count' and containing 'segment_marker'.
        Set the value without the removed segments as environment variables instead of the previous one.

        :return: original 'env_name' value, for possibility of revert.
        """
        original_path = os.environ.get(env_name, None)
        if not original_path:
            return original_path

        path_segments = original_path.split(os.pathsep)
        new_path = original_path
        for i in range(min(count, len(path_segments))):
            segment = path_segments[i]
            if segment_marker in segment:
                new_path = new_path.replace(segment + os.pathsep, '', 1)

        os.environ[env_name] = new_path
        return original_path


    def new_function(*args, **kwargs):
        """
        Wrapper function for Linux TVB altered environment variables.
        Apply this only on Linux, as that is the only env in which TVB start scripts alter LD_*_PATH env vars.

        :return: Result of the wrapped function
        """
        if not Environment().is_linux_deployment():
            # Do nothing
            return func(*args, **kwargs)

        original_ld_library_path = _remove_from_path('LD_LIBRARY_PATH', 2, 'tvb_data')
        original_ld_run_path = _remove_from_path('LD_RUN_PATH', 2, 'tvb_data')

        result = func(*args, **kwargs)
        ## Restore environment settings after function executed.
        if original_ld_library_path:
            os.environ['LD_LIBRARY_PATH'] = original_ld_library_path
        if original_ld_run_path:
            os.environ['LD_RUN_PATH'] = original_ld_run_path
        return result

    return new_function

