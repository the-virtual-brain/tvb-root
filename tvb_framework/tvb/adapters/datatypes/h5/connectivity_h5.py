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

from tvb.basic.neotraits.api import NArray
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Json, STORE_STRING, MEMORY_STRING
from tvb.datatypes.connectivity import Connectivity


class ConnectivityH5(H5File):
    def __init__(self, path):
        super(ConnectivityH5, self).__init__(path)
        self.region_labels = DataSet(NArray(dtype=STORE_STRING), self, "region_labels")
        self.weights = DataSet(Connectivity.weights, self)
        self.undirected = Scalar(Connectivity.undirected, self)
        self.tract_lengths = DataSet(Connectivity.tract_lengths, self)
        self.centres = DataSet(Connectivity.centres, self)
        self.cortical = DataSet(Connectivity.cortical, self)
        self.hemispheres = DataSet(Connectivity.hemispheres, self)
        self.orientations = DataSet(Connectivity.orientations, self)
        self.areas = DataSet(Connectivity.areas, self)
        self.number_of_regions = Scalar(Connectivity.number_of_regions, self)
        self.number_of_connections = Scalar(Connectivity.number_of_connections, self)
        self.parent_connectivity = Scalar(Connectivity.parent_connectivity, self)
        self.saved_selection = Json(Connectivity.saved_selection, self)

    def get_centres(self):
        return self.centres.load()

    def get_region_labels(self):
        return self.region_labels.load()

    def store(self, datatype, scalars_only=False, store_references=True):
        # type: (Connectivity, bool, bool) -> None
        super(ConnectivityH5, self).store(datatype, scalars_only, store_references)
        self.region_labels.store(datatype.region_labels.astype(STORE_STRING))

    def load_into(self, datatype):
        # type: (Connectivity) -> None
        super(ConnectivityH5, self).load_into(datatype)
        datatype.region_labels = self.region_labels.load().astype(MEMORY_STRING)
