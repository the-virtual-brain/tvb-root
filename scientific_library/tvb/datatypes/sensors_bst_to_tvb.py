# -*- coding: utf-8 -*-
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Small script for converting Brainstorm sensor files for our default dataset
to the simple ASCII format used by TVB (and other software).

NB: Brainstorm uses meters, TVB uses millimeters.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy
import scipy.io


def get_field_array(mat_group, n=3, dtype=numpy.float64):
    return numpy.array([
        l.flat[:n] if l.size else numpy.zeros((n, ), dtype)
        for l in mat_group[0]
    ], dtype=dtype)

def convert_brainstorm_to_tvb(tvb_data_path, chan_paths):
    """
    Convert given set of channels from Brainstorm to TVB formats.

    """

    bst_path = tvb_data_path + 'brainstorm/data/TVB-Subject/'
    for sens_type, sens_path in chan_paths.items():
        # only MEG channels require orientation information
        use_ori = sens_type in ('meg', )
        # read from MAT file necessary fields
        mat = scipy.io.loadmat(bst_path + sens_path)
        name = [l[0] for l in mat['Channel']['Name'][0]]
        loc = get_field_array(mat['Channel']['Loc'])
        if use_ori:
            ori = get_field_array(mat['Channel']['Orient'])
        # bst uses m, we use mm
        loc *= 1e3
        # write out to text format
        out_fname = '%s/sensors/%s-brainstorm-%d.txt'
        out_fname %= tvb_data_path, sens_type, len(name)
        with open(out_fname, 'w') as fd:
            if use_ori: # MEG
                for n, (x, y, z), (ox, oy, oz) in zip(name, loc, ori):
                    line = '\t'.join(['%s']+['%f']*6) + '\n'
                    line %= n, x, y, z, ox, oy, oz
                    fd.write(line)
            else: # sEEG, EEG
                for n, (x, y, z) in zip(name, loc):
                    line = '\t'.join(['%s']+['%f']*3) + '\n'
                    line %= n, x, y, z
                    fd.write(line)


if __name__ == '__main__':
    import tvb_data
    convert_brainstorm_to_tvb(tvb_data.__path__,
        chan_paths={
            'eeg': 'EEG_channels/channel.mat',
            'meg': 'MEGchannels/channel_4d_acc1.mat',
            'seeg': 'seeg_channels/channel.mat',
        })