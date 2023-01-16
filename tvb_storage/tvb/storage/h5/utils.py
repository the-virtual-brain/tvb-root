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

from datetime import datetime

COMPLEX_TIME_FORMAT = '%Y-%m-%d,%H-%M-%S.%f'
# LESS_COMPLEX_TIME_FORMAT is also compatible with data exported from TVB 1.0.
# This is only used as a fallback in the string to date conversion.
LESS_COMPLEX_TIME_FORMAT = '%Y-%m-%d,%H-%M-%S'
SIMPLE_TIME_FORMAT = "%m-%d-%Y"


def string2date(string_input, complex_format=True, date_format=None):
    """Read date from string, after internal format"""
    if string_input == 'None':
        return None
    if date_format is not None:
        return datetime.strptime(string_input, date_format)
    if complex_format:
        try:
            return datetime.strptime(string_input, COMPLEX_TIME_FORMAT)
        except ValueError:
            # For backwards compatibility with TVB 1.0
            return datetime.strptime(string_input, LESS_COMPLEX_TIME_FORMAT)
    return datetime.strptime(string_input, SIMPLE_TIME_FORMAT)


def date2string(date_input, complex_format=True, date_format=None):
    """Convert date into string, after internal format"""
    if date_input is None:
        return "None"

    if date_format is not None:
        return date_input.strftime(date_format)

    if complex_format:
        return date_input.strftime(COMPLEX_TIME_FORMAT)
    return date_input.strftime(SIMPLE_TIME_FORMAT)


def string2bool(string_input):
    """ Convert given string into boolean value."""
    string_input = str(string_input).lower()
    return string_input in ("yes", "true", "t", "1")

