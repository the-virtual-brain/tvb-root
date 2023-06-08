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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.basic.exceptions import TVBException


class InvalidFilterChainInput(TVBException):
    """
    Exception raised if a input that does not match the declared filter is passed as input.
    """

    def __init__(self, message):
        TVBException.__init__(self, message)


class InvalidFilterEntity(TVBException):
    """
    Exception raised if a filter entity has contradicting values for operations and values.
    for example: in None, not in 1 
    """

    def __init__(self, message):
        TVBException.__init__(self, message)
