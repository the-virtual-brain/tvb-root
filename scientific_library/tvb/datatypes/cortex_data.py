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
import numpy
from tvb.basic.traits import types_basic as basic, core
from tvb.datatypes import arrays as arrays
from tvb.datatypes.local_connectivity_data import LocalConnectivityData
from tvb.datatypes.region_mapping_data import RegionMappingData
from tvb.datatypes.surfaces import CorticalSurface


class CortexData(CorticalSurface):
    """
    Extends the CorticalSurface class to add specific attributes of the cortex,
    such as, local connectivity...
    """
    _ui_complex_datatype = CorticalSurface

    _ui_name = "A cortex..."

    local_connectivity = LocalConnectivityData(label="Local Connectivity",
                                               required=False,
                                               doc="""Define the interaction
                                               between neighboring network nodes.
                                               This is implicitly integrated in
                                               the definition of a given surface
                                               as an excitatory mean coupling of
                                               directly adjacent neighbors to
                                               the first state variable of each
                                               population model (since these
                                               typically represent the mean-neural
                                               membrane voltage).
                                               This coupling is instantaneous
                                               (no time delays).""")

    region_mapping_data = RegionMappingData(
        label="region mapping",
        doc="""An index vector of length equal to the number_of_vertices + the
            number of non-cortical regions, with values that index into an
            associated connectivity matrix.""")  # 'CS'

    coupling_strength = arrays.FloatArray(
        label="Local coupling strength",
        range=basic.Range(lo=0.0, hi=20.0, step=1.0),
        default=numpy.array([1.0]),
        file_storage=core.FILE_STORAGE_NONE,
        doc="""A factor that rescales local connectivity strengths.""")

    eeg_projection = arrays.FloatArray(
        label="EEG projection", order=-1,
        #NOTE: This is redundant if the EEG monitor isn't used, but it makes life simpler.
        required=False,
        doc="""A 2-D array which projects the neural activity on the cortical
            surface to a set of EEG sensors.""")
    #  requires linked sensors.SensorsEEG and Skull/Skin/Air

    meg_projection = arrays.FloatArray(
        label="MEG projection",
        #linked = ?sensors, skull, skin, etc?
        doc="""A 2-D array which projects the neural activity on the cortical
            surface to a set of MEG sensors.""",
        required=False, order=-1,)
    #  requires linked SensorsMEG and Skull/Skin/Air


    internal_projection = arrays.FloatArray(
        label="Internal projection",
        required=False, order=-1,
        #linked = ?sensors, skull, skin, etc?
        doc="""A 2-D array which projects the neural activity on the
            cortical surface to a set of embeded sensors.""")
    #  requires linked SensorsInternal

    __generate_table__ = False



    def populate_cortex(self, cortex_surface, cortex_parameters=None):
        """
        Populate 'self' from a CorticalSurfaceData instance with additional
        CortexData specific attributes.

        :param cortex_surface:  CorticalSurfaceData instance
        :param cortex_parameters: dictionary key:value, where key is attribute on CortexData
        """
        for name in cortex_surface.trait:
            try:
                setattr(self, name, getattr(cortex_surface, name))
            except Exception, exc:
                self.logger.exception(exc)
                self.logger.error("Could not set attribute '" + name + "' on Cortex")
        for key, value in cortex_parameters.iteritems():
            setattr(self, key, value)
        return self


    @property
    def region_mapping(self):
        """
        Define shortcut for retrieving RegionMapping map array.
        """
        if self.region_mapping_data is None:
            return None
        return self.region_mapping_data.array_data