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
Define DB changes for each release.

.. moduleauthor:: X Y <x.y@codemart.ro>

EXAMPLE FILE:

from sqlalchemy import Column, String
from migrate.changeset.schema import create_column, drop_column
from tvb.core.entities import model

COL_1 = Column('orientations_path', String)
COL_2 = Column('areas_path', String)

COL_3 = Column("local_connectivity_path", String)

COL_4 = Column("uid", String)
meta = model.Base.metadata
    

def upgrade(migrate_engine):
    "
    Upgrade operations go here. 
    Don't create your own engine; bind migrate_engine to your metadata.
    "
    meta.bind = migrate_engine
    table1 = meta.tables['DATA_connectivity']
    create_column(COL_1, table1)
    create_column(COL_2, table1)

    table2 = meta.tables['DATA_measuredcortex']
    create_column(COL_3, table2)
    
    table3 = meta.tables['DATA_TYPES']
    create_column(COL_4, table3)
    

def downgrade(migrate_engine):
    "
    Operations to reverse the above upgrade go here.
    "
    meta.bind = migrate_engine
    table1 = meta.tables['DATA_connectivity']
    table2 = meta.tables['DATA_measuredcortex']
    drop_column(COL_1, table1)
    drop_column(COL_2, table1)
    drop_column(COL_3, table2)
    
"""    
    
    