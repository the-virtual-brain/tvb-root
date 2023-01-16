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
This is a framework helper module.
Some framework datatypes have functions that will be called via http by the TVB GUI.
These functions will receive some arguments as strings and return json serializable structures usually dicts.
This module contains functions to parse those strings and construct those responses.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy
from tvb.basic import exceptions



def parse_slice(slice_string):
    """
    Parse a slicing expression
    >>> parse_slice("::1, :")
    (slice(None, None, 1), slice(None, None, None))
    >>> parse_slice("2")
    2
    >>> parse_slice("[2]")
    2
    """
    ret = []
    slice_string = slice_string.replace(' ', '')

    # remove surrounding brackets if any
    if slice_string[0] == '[' and slice_string[-1] == ']':
        slice_string = slice_string[1:-1]

    for d in slice_string.split(','):
        frags = d.split(':')
        if len(frags) == 1:
            if frags[0] == '':
                ret.append(slice(None))
            else:
                ret.append(int(frags[0]))
        elif len(frags) <= 3:
            frags = [int(d) if d else None for d in frags]
            ret.append(slice(*frags))
        else:
            raise ValueError('invalid slice')
    if len(ret) > 1:
        return tuple(ret)
    else:
        return ret[0]



def slice_str(slice_or_tuple):
    """
    >>> slice_str(slice(1, None, 2))
    '1::2'
    >>> slice_str((slice(None, None, 2), slice(None), 4))
    '::2, :, 4'
    Does not handle ... yet
    """
    def sl_str(s):
        if isinstance(s, slice):
            if s.start is s.step is None:
                return '%s' % (s.stop if s.stop is not None else ':')
            r = '%s:%s' % (s.start or '', s.stop or '')
            if s.step is not None:
                r += ':%d' % s.step
            return r
        else:
            return str(int(s))

    if isinstance(slice_or_tuple, tuple):
        return '[' + ', '.join(sl_str(s) for s in slice_or_tuple) + ']'
    else:
        return '[' + sl_str(slice_or_tuple) + ']'



def preprocess_space_parameters(x, y, z, max_x, max_y, max_z):
    """
    Covert ajax call parameters into numbers and validate them.

    :param x:  coordinate
    :param y:  coordinate
    :param z:  coordinate that will be reversed
    :param max_x: maximum x accepted value
    :param max_y: maximum y accepted value
    :param max_z: maximum z accepted value

    :return: (x, y, z) as integers, Z reversed
    """

    x, y, z = int(x), int(y), int(z)

    if not 0 <= x <= max_x or not 0 <= y <= max_y or not 0 <= z <= max_z:
        msg = "Coordinates out of boundaries: [x,y,z] = [{0}, {1}, {2}]".format(x, y, z)
        raise exceptions.ValidationException(msg)

    # Reverse Z
    z = max_z - z - 1

    return x, y, z



def preprocess_time_parameters(t1, t2, time_length):
    """
    Covert ajax call parameters into numbers and validate them.

    :param t1: start time
    :param t2: end time
    :param time_length: maximum time length in current TS

    :return: (t1, t2, t2-t1) as numbers
    """

    from_idx = int(t1)
    to_idx = int(t2)

    if not 0 <= from_idx < to_idx <= time_length:
        msg = "Time indexes out of boundaries: from {0} to {1}".format(from_idx, to_idx)
        raise exceptions.ValidationException(msg)

    current_time_line = max(to_idx - from_idx, 1)

    return from_idx, to_idx, current_time_line


def postprocess_voxel_ts(ts, slices, background_value=None, background_min=None, background_max=None, label=None):
    """
    Read TimeLine from TS and prepare the result for TSVolumeViewer.

    :param ts: TS instance, with read_data_slice method
    :param slices: slices for reading from H5

    :return: A complex dictionary with information about current voxel.
    """

    if background_value is not None:
        time_line = background_value
    else:
        time_line = ts.read_data_slice(slices).flatten()


    result = dict(data=time_line.tolist(),
                  min=background_min or float(min(time_line)),
                  max=background_max or float(max(time_line)),
                  mean=float(numpy.mean(time_line)),
                  median=float(numpy.median(time_line)),
                  variance=float(numpy.var(time_line)),
                  deviation=float(numpy.std(time_line)),
                  label=label)
    return result