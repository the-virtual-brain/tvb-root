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
This is the root of TVB tests. It overlaps between tvb-library and tvb-framework packages.

For running test use the command: pytest [tests_folder]

IMPORTANT: To ensure the correct TVB Profile is set in tests, before ANY tvb import in the unit test,
    setup the correct tvb test profile::

        from tvb.tests.library import setup_test_console_env
        setup_test_console_env()

        # OR
        from tvb.tests.framework.core.base_testcase import init_test_env
        init_test_env()

    You can do this implicitly (as done currently in the majority of our example unit tests,
    by importing BaseTestCase FIRST::

        from tvb.tests.library.base_testcase import BaseTestCase
        # OR
        from tvb.tests.framework.core.base_testcase import BaseTestCase
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
