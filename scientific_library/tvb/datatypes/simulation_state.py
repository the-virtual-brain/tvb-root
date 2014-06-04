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
DataType for storing a simulator's state in files and as DB reference.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""


import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays
from tvb.basic.traits.types_mapped import MappedType


class SimulationState(MappedType):
    """
    Simulation State, prepared for H5 file storage.
    """  
    
    # History Array
    history = arrays.FloatArray(required=False)
    # Simulator step number
    current_step = basic.Integer()
    # Array with _stock array for every monitor configured in current simulation.
    # As the monitors are dynamic, we prepare a bunch of arrays for storage in H5 file.
    monitor_stock_1 = arrays.FloatArray(required=False)
    monitor_stock_2 = arrays.FloatArray(required=False)
    monitor_stock_3 = arrays.FloatArray(required=False)
    monitor_stock_4 = arrays.FloatArray(required=False)
    monitor_stock_5 = arrays.FloatArray(required=False)
    monitor_stock_6 = arrays.FloatArray(required=False)
    monitor_stock_7 = arrays.FloatArray(required=False)
    monitor_stock_8 = arrays.FloatArray(required=False)
    monitor_stock_9 = arrays.FloatArray(required=False)
    monitor_stock_10 = arrays.FloatArray(required=False)
    monitor_stock_11 = arrays.FloatArray(required=False)
    monitor_stock_12 = arrays.FloatArray(required=False)
    monitor_stock_13 = arrays.FloatArray(required=False)
    monitor_stock_14 = arrays.FloatArray(required=False)
    monitor_stock_15 = arrays.FloatArray(required=False)
    
    
    def __init__(self, **kwargs):
        """ 
        Constructor for Simulator State
        """
        super(SimulationState, self).__init__(**kwargs)
        self.visible = False
    
    
    def populate_from(self, simulator_algorithm):
        """
        Prepare a state for storage from a Simulator object.
        """
        self.history = simulator_algorithm.history
        self.current_step = simulator_algorithm.current_step
        
        i = 1
        for monitor in simulator_algorithm.monitors:
            field_name = "monitor_stock_" + str(i)
            setattr(self, field_name, monitor._stock)
            if hasattr(monitor, "_ui_name"):
                self.set_metadata({'monitor_name': monitor._ui_name}, field_name)
            else:
                self.set_metadata({'monitor_name': monitor.__class__.__name__}, field_name)
            i = i + 1
        
    
    def fill_into(self, simulator_algorithm):
        """
        Populate a Simulator object from current stored-state.
        """
        simulator_algorithm.history = self.history 
        simulator_algorithm.current_step = self.current_step
        
        i = 1
        for monitor in simulator_algorithm.monitors:
            monitor._stock = getattr(self, "monitor_stock_" + str(i))
            i = i + 1
  
    
 