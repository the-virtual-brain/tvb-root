# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
converts a color scheme texture image to json arrays
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import Image
import numpy

def color_texture_to_list(img_pth, img_width, band_height, bands_number):
    """
    :param img_pth: Path to the texure
    :param img_width: Texture width
    :param band_height: Height of a color scheme band
    :param bands_number: The number of bands to serialize
    :return: A list of bands_number color schemes. A scheme is a list of img_width colors
    """
    im = Image.open(img_pth)
    ima = numpy.asarray(im)

    if ima.shape != (img_width, img_width, 4):
        raise ValueError("unexpected image shape " + str(ima.shape))
    tex_vs = [(i * band_height + 0.5)/img_width for i in xrange(bands_number)]
    color_schemes = []
    for v in tex_vs:
        idx = int(v * img_width)
        color_schemes.append(ima[idx, :, :3].tolist())
    return color_schemes