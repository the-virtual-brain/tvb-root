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
Mayavi visualisation script for cortical surface
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>
"""

import numpy as np
from mayavi import mlab

if __name__ == '__main__':
    vtx_lh = np.loadtxt('vertices_cortex_ph0036pial_lh.txt.bz2')
    tri_lh = np.loadtxt('triangles_cortex_ph0036pial_lh.txt.bz2')

    vtx_rh = np.loadtxt('vertices_cortex_ph0036pial_rh.txt.bz2')
    tri_rh = np.loadtxt('triangles_cortex_ph0036pial_rh.txt.bz2')
    fig = mlab.figure(figure='surface', fgcolor=(0.5, 0.5, 0.5))

    mlab.triangular_mesh(vtx_lh[:, 0], vtx_lh[:, 1], vtx_lh[:, 2], tri_lh,
                         color=(0.7, 0.67, 0.67),
                         representation = 'surface',
                         figure=fig)

    mlab.triangular_mesh(vtx_rh[:, 0], vtx_rh[:, 1], vtx_rh[:, 2], tri_rh,
                         color=(0.7, 0.67, 0.67),
                         representation = 'surface',
                         figure=fig)

    mlab.show()