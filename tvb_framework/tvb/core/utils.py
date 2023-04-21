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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import json
from datetime import datetime
import uuid
import urllib.request, urllib.parse, urllib.error
from hashlib import md5
import numpy
import six
from tvb.basic.profile import TvbProfile

CHAR_SEPARATOR = "__"
CHAR_SPACE = "--"
CHAR_DRIVE = "-DriVe-"
DRIVE_SEP = ":"

COMPLEX_TIME_FORMAT = '%Y-%m-%d,%H-%M-%S.%f'
# LESS_COMPLEX_TIME_FORMAT is also compatible with data exported from TVB 1.0.
# This is only used as a fallback in the string to date conversion.
LESS_COMPLEX_TIME_FORMAT = '%Y-%m-%d,%H-%M-%S'
SIMPLE_TIME_FORMAT = "%m-%d-%Y"


################## PATH related methods start here ###############

def path2url_part(file_path):
    """
    Prepare a File System Path for passing into an URL.
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(TvbProfile.current.TVB_STORAGE, file_path)
    result = file_path.replace(os.sep, CHAR_SEPARATOR).replace(" ", CHAR_SPACE).replace(DRIVE_SEP, CHAR_DRIVE)
    return urllib.parse.quote(result)


def url2path(encoded_path):
    """
    Retrieve File System Path from encoded URL (inverse of path2url_part).
    """
    return encoded_path.replace(CHAR_SEPARATOR, os.sep).replace(CHAR_SPACE, " ").replace(CHAR_DRIVE, DRIVE_SEP)


def get_unique_file_name(storage_folder, file_name, try_number=0):
    """
    Compute non-existent file name, in storage_folder.
    Try file_name, and if already exists, try adding a number.
    """
    # TODO this method should be re-tought
    name, ext = os.path.splitext(file_name)
    date = str(datetime.now())
    date = date.replace(' ', '').replace(':', '').replace('.', '').replace('-', '')
    if try_number > 0:
        file_ = '%s-%s%s' % (name, date, ext)
    else:
        file_ = file_name
    full_path = os.path.join(storage_folder, file_)
    if os.path.exists(full_path):
        # Try another name, by appending the consecutive try_number
        return get_unique_file_name(storage_folder, file_name, try_number + 1)
    return full_path, file_


################## PATH related methods end here ###############

################## CONVERT related methods start here ###############

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


def parse_json_parameters(parameters):
    """
    From JSON with Unicodes, return a dictionary having strings as keys.
    Loading from DB a JSON will return instead of string keys, unicodes.
    """
    params = json.loads(parameters)
    new_params = {}
    for key, value in six.iteritems(params):
        new_params[str(key)] = value
    return new_params


def format_timedelta(timedelta, most_significant2=True):
    """
    Format a datetime.timedelta.
    :param timedelta: object timedelta to format
    :param most_significant2: Will show only the 2 most significant units (ex: hours, minutes). Default True.
    """
    days = timedelta.days
    hours, remainder = divmod(timedelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    fragments = [str(days) + 'd', str(hours) + 'h', str(minutes) + 'm', str(seconds) + 's']

    if days:
        fragments = fragments[0:]
    elif hours:
        fragments = fragments[1:]
    elif minutes:
        fragments = fragments[2:]
    else:
        fragments = fragments[3:]

    if most_significant2:
        fragments = fragments[:2]

    return ' '.join(fragments)


class TVBJSONEncoder(json.JSONEncoder):
    """
    Custom encoder class. Referring towards "to_json" method, when found, or default behaviour otherwise.
    """

    def default(self, obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        try:
            # TVB-2565 numpy int serialization
            if numpy.issubdtype(obj, numpy.integer):
                return int(obj)
        except TypeError:
            pass
        return json.JSONEncoder.default(self, obj)

################## CONVERT related methods end here ###############

################## GENERIC  methods start here ###############

def generate_guid():
    """ 
    Generate new Global Unique Identifier.
    This identifier should be unique per each station, 
    and unique for different machines.
    """
    return str(uuid.uuid1())


def format_bytes_human(size, si=False):
    """
    :param size: size in kilobytes
    :param si: if True use SI units (multiple of 1000 not 1024)
    :return: a String with [number] [memory unit measure]
    """
    if si:
        m = ['kB', 'MB', 'GB']
        base = 1000.0
    else:
        m = ['KiB', 'MiB', 'GiB']
        base = 1024.0

    exp = 0
    while size >= base and exp < len(m) - 1:
        size /= base
        exp += 1
    return "%.1f %s" % (size, m[exp])


def prepare_time_slice(total_time_length, max_length=10 ** 4):
    """
    Limit the time dimension when retrieving from TS.
    If total time length is greater than MAX, then retrieve only the last part of the TS

    :param total_time_length: TS time dimension
    :param max_length: limiting number of TS steps

    :return: python slice
    """

    if total_time_length < max_length:
        return slice(total_time_length)

    return slice(total_time_length - max_length, total_time_length)


def hash_password(pass_string):
    return md5(pass_string.encode('utf-8')).hexdigest()
