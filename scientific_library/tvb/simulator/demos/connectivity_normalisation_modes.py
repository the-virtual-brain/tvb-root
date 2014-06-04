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
Demonstrate the effect of different normalisation ``modes`` on the connectivity
strength range. 

Current modes are re-scaling methods.

.. moduleauthor:: Paula Sanz Leon <pau.sleon@gmail.com>

"""

from tvb.simulator.lab import *

LOG.info("Reading default connectivity...")
white_matter = connectivity.Connectivity(load_default=True)
white_matter.configure()
con = connectivity.Connectivity(load_default=True)
con.configure()

#scale weights by the maximum absolute value
con.weights = white_matter.scaled_weights(mode='tract')
plot_connectivity(con, num="tract_mode", plot_tracts=False)

#undo scaling
con.weights = white_matter.scaled_weights(mode='none')
plot_connectivity(con, num="default_mode", plot_tracts=False)

#re-scale using another `` mode``
con.weights = white_matter.scaled_weights(mode='region')
plot_connectivity(con, num="region_mode", plot_tracts=False)

#undo scaling
con.weights = white_matter.scaled_weights(mode='none')
plot_connectivity(con, num="default_mode", plot_tracts=False)

#binarize
con.weights = white_matter.transform_binarize_matrix()
plot_connectivity(con, num="default_mode", plot_tracts=False)

#remove-self connections 
con.weights = white_matter.transform_remove_self_connections()
plot_connectivity(con, num="default_mode", plot_tracts=False)

pyplot.show()





