# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.connectivity import Connectivity as TVBConnectivity


class Connectivity(TVBConnectivity, BaseModel):

    # The following two methods help avoid problems with centers vs centres writing
    def __setattr__(self, key, value):
        if key == "centers":
            super(Connectivity, self).__setattr__("centres", value)
        else:
            super(Connectivity, self).__setattr__(key, value)

    @property
    def centers(self):
        return self.centres

    # A usefull method for addressing subsets of the connectome by label:
    def get_regions_inds_by_labels(self, labels):
        return self.labels2inds(self.region_labels, labels)

    def to_tvb_instance(self, **kwargs):
        return super(Connectivity, self).to_tvb_instance(TVBConnectivity, **kwargs)
