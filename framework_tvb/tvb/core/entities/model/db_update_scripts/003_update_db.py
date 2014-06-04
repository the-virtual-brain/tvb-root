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
Change of DB structure to TVB 1.0.3.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy.sql import text
from sqlalchemy import Table, MetaData, Column, Integer, String
from migrate.changeset.schema import create_column
from tvb.core.entities.storage import SA_SESSIONMAKER


COL_1 = Column('_number_of_values', Integer)
COL_2 = Column('_invdx', String)
COL_3 = Column('_xmax', String)
COL_4 = Column('_df', String)
COL_5 = Column('_dx', String)
COL_6 = Column('_xmin', String)
COL_7 = Column('_data', String)


def _prepare_table(meta, table_name):
    """
    Load current DB structure.
    """
    table = Table(table_name, meta, autoload=True)
    for constraint in table._sorted_constraints:
        if not isinstance(constraint.name, (str, unicode)):
            constraint.name = None
    return table


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    meta = MetaData(bind=migrate_engine)
    
    table = _prepare_table(meta, 'USER_PREFERENCES')
    table.c.user_id.alter(name='fk_user')
    
    table = _prepare_table(meta, 'BURST_CONFIGURATIONS')
    table.c.project_id.alter(name='fk_project')
    
    table = _prepare_table(meta, 'WORKFLOWS',)
    table.c.project_id.alter(name='fk_project')
    table.c.burst_id.alter(name='fk_burst')
    
    table = _prepare_table(meta, 'WORKFLOW_STEPS')
    table.c.workflow_id.alter(name='fk_workflow')
    table.c.algorithm_id.alter(name='fk_algorithm')
    table.c.resulted_op_id.alter(name='fk_operation')
    
    table = _prepare_table(meta, 'MAPPED_DATATYPE_MEASURE')
    table.c.analyzed_datatype.alter(name='_analyzed_datatype')
    
    ## Fix Lookup Table mapping.
    table = _prepare_table(meta, 'MAPPED_LOOK_UP_TABLE_DATA')
    create_column(COL_1, table)
    create_column(COL_2, table)
    create_column(COL_3, table)
    create_column(COL_4, table)
    create_column(COL_5, table)
    create_column(COL_6, table)
    create_column(COL_7, table)
    session = SA_SESSIONMAKER()
    session.execute(text('DELETE FROM "MAPPED_LOOK_UP_TABLE_DATA";'))
    session.execute(text('insert into "MAPPED_LOOK_UP_TABLE_DATA"(id, _equation, _number_of_values, _invdx, _xmax, _xmin, _df, _dx, _data) '
                         'select id, \'\', _number_of_values, _invdx, _xmax, _xmin, _df, _dx, _data from "MAPPED_NERF_TABLE_DATA";'))
    session.execute(text('insert into "MAPPED_LOOK_UP_TABLE_DATA"(id, _equation, _number_of_values, _invdx, _xmax, _xmin, _df, _dx, _data) '
                         'select id, \'\', _number_of_values, _invdx, _xmax, _xmin, _df, _dx, _data from "MAPPED_PSI_TABLE_DATA";'))
    session.commit()
    session.close()
        
    table = _prepare_table(meta, 'MAPPED_NERF_TABLE_DATA')
    table.drop()
    
    table = _prepare_table(meta, 'MAPPED_PSI_TABLE_DATA')
    table.drop()



def downgrade(migrate_engine):
    """
    Operations to reverse the above upgrade go in this function.
    """
    meta = MetaData(bind=migrate_engine)
    
    table = _prepare_table(meta, 'USER_PREFERENCES')
    table.c.fk_user.alter(name='user_id')
    
    table = _prepare_table(meta, 'BURST_CONFIGURATIONS')
    table.c.fk_project.alter(name='project_id')
    
    table = _prepare_table(meta, 'WORKFLOWS')
    table.c.fk_project.alter(name='project_id')
    table.c.fk_burst.alter(name='burst_id')
    
    table = _prepare_table(meta, 'WORKFLOW_STEPS')
    table.c.fk_workflow.alter(name='workflow_id')
    table.c.fk_algorithm.alter(name='algorithm_id')
    table.c.fk_operation.alter(name='resulted_op_id')
    
    table = _prepare_table(meta, 'MAPPED_DATATYPE_MEASURE')
    table.c._analyzed_datatype.alter(name='analyzed_datatype') 
    
    
        