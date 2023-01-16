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
Converts a color scheme texture image to json arrays

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""


import numpy
from PIL import Image

def color_texture_to_list(img_pth, img_width, band_height):
    """
    :param img_pth: Path to the texure
    :param img_width: Texture width
    :param band_height: Height of a color scheme band
    :return: A list of img_width/band_height color schemes. A scheme is a list of img_width colors
    """
    im = Image.open(img_pth)
    ima = numpy.asarray(im)

    if ima.shape != (img_width, img_width, 4):
        raise ValueError("unexpected image shape " + str(ima.shape))
    tex_vs = [(i * band_height + 0.5)/img_width for i in range(img_width//band_height)]
    color_schemes = []
    for v in tex_vs:
        idx = int(v * img_width)
        color_schemes.append(ima[idx, :, :3].tolist())
    return color_schemes