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
Provide some preprocessing utilities.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import subprocess


# need a Docker image + some data to test these parts automatically


class FreeSurfer:
    "Interface to a few useful FreeSurfer commands."

    def __init__(self, freesurfer_home=None, subjects_dir=None):
        self.freesurfer_home = freesurfer_home or self.guess_freesurfer_home()
        self.subjects_dir = subjects_dir or self.guess_subjects_dir()

    def recon_all(self, subject, input_image):
        "Run the recon-all pipeline."
        cmd = ['recon-all', '-all', '-i', input_image, '-s', subject]
        subprocess.check_call(cmd)

    def resamp_anat(self, subject, target_resolution='fsaverage5'):
        "Resample cortical surfaces at lower resolution."
        raise NotImplemented


class FSL:
    "Interface to a few useful FSL commands."

    # fslreorient2std


class Mrtrix:
    "Interface to a few useful Mrtrix commands."

    # mrinfo
    # mrconvert
    # dwipreproc
    # dwi2mask
    # dwi2response
    # dwi2fod
    # tckgen
    # tcksift?
    # labelgen
    # tck2connectome


class OpenMEEG:
    "Interface to a few useful OpenMEEG commands."

    # generate geometry & other files from sensor locations etc
    # with FreeSurfer cortex
    # construct model
    # construct gain matrices
