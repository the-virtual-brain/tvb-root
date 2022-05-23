# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
