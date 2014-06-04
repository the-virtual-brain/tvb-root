# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

A subject's dataset


* volume data -> MRI acquisition -> Registration -> Coordinates transform to MNI space 
 |
  \-- voxel-based gray matter parcellation (obtain parcellation mask) -> AAL/anatomical template
 
* surfaces data (cortical, skull, skin surfaces extraction) -> FSL/BET 

* connectivity data (white matter weights, tract-lengths)   -> Diffusion Toolkit + TrackVis

* region mapping between parcellation and number of vertices in the cortical surface.

+ lead-field matrices (ie, projection matrices) mapping nodes onto EEG/MEG space
        
.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>

"""

from tvb.simulator.lab import *

# From the inside out
connectome       = connectivity.Connectivity(load_default=True)
cortical_surface = surfaces.Cortex.from_file()
brain_skull      = surfaces.BrainSkull(load_default=True)
skull_skin       = surfaces.SkullSkin(load_default=True)
skin_air		 = surfaces.SkinAir(load_default=True)


# Get info
centres = connectome.centres

try:
    from tvb.simulator.plot.tools import mlab
    fig_tvb = mlab.figure(figure='John Doe', bgcolor=(0.0, 0.0, 0.0))
    
    region_centres = mlab.points3d(centres[:, 0], 
                                   centres[:, 1], 
                                   centres[:, 2],
                                   color=(1.0, 0.0, 0.0),
                                   scale_factor = 7.,
                                   figure = fig_tvb)
    
    
    plot_surface(cortical_surface, fig=fig_tvb, op=0.9, rep='fancymesh')
    plot_surface(brain_skull, fig=fig_tvb, op=0.2)
    plot_surface(skull_skin, fig=fig_tvb, op=0.15)
    plot_surface(skin_air, fig=fig_tvb, op=0.1)
    
    # Plot them
    mlab.show(stop=True)
except ImportError:

    LOG.exception("Could not display!")
    pass

#EoF