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
Change of DB structure from TVB version 1.0 to TVB 1.0.1.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.core.entities.storage import dao


def upgrade(_migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    simulator_db_group = dao.find_group('tvb.adapters.simulator.simulatorAdapter', 'SimulatorAdapter')
    if simulator_db_group is not None:
        simulator_db_group.module = 'tvb.adapters.simulator.simulator_adapter'
        simulator_db_group.classname = 'SimulatorAdapter'
        dao.store_entity(simulator_db_group)
    

def downgrade(_migrate_engine):
    """Operations to reverse the above upgrade go here."""
    simulator_db_group = dao.find_group('tvb.adapters.simulator.simulator_adapter', 'SimulatorAdapter')
    if simulator_db_group is not None:
        simulator_db_group.module = 'tvb.adapters.simulator.simulatorAdapter'
        simulator_db_group.classname = 'SimulatorAdapter'
        dao.store_entity(simulator_db_group)
        
        
        