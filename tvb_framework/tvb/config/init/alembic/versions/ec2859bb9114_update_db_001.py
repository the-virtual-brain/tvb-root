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

"""
Revision ID: ec2859bb9114
Create Date: 2021-01-29
"""

from alembic import op
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import Column, String, Integer
import uuid
import json
from sqlalchemy.sql.schema import UniqueConstraint
from tvb.basic.profile import TvbProfile
from tvb.core.neotraits.db import Base
from tvb.basic.logger.builder import get_logger

# revision identifiers, used by Alembic.
revision = 'ec2859bb9114'
down_revision = None
conn = op.get_bind()
LOGGER = get_logger(__name__)


def _migrate_range_params(ranges):
    new_ranges = []
    for range in ranges:
        list_range = eval(range)

        if list_range is None:
            new_ranges.append('None')
            continue

        # in the range param name if the range param is not a gid param then
        # all the characters between the first and last underscore (including them)
        # must be deleted and replaced with a dot

        param_name = list_range[0]
        param_range = list_range[1]
        if '_' in param_name:
            if param_name.count('_') > 1:
                first_us = param_name.index('_')
                last_us = param_name.rfind('_')
                string_to_be_replaced = param_name[first_us:last_us + 1]
                param_name = param_name.replace(string_to_be_replaced, '.')

            param_name = "\"" + param_name + "\""

            # in the old version the range was a list of all values that the param had, but in the new one we
            # need only the minimum, maximum and step value
            range_dict = dict()
            range_dict['\"lo\"'] = param_range[0]
            range_dict['\"hi\"'] = param_range[-1]
            range_dict['\"step\"'] = param_range[1] - param_range[0]
            new_ranges.append([param_name, range_dict])
        else:
            # We have a gid param
            new_ranges.append(['\"' + param_name + '\"', 'null'])
    return new_ranges


def _update_range_parameters(burst_config_table, operation_groups_table, range1, range2, op_group_id):

    conn.execute(burst_config_table.update().
                 where(burst_config_table.c.fk_operation_group == op_group_id).
                 values({'range1': range1}))

    conn.execute(operation_groups_table.update().
                 where(operation_groups_table.c.id == op_group_id).
                 values({'range1': range1}))

    if range2 != 'None':
        conn.execute(burst_config_table.update().
                     where(burst_config_table.c.fk_operation_group == op_group_id).
                     values({'range2': range2}))

        conn.execute(operation_groups_table.update().
                     where(operation_groups_table.c.id == op_group_id).
                     values({'range2': range2}))


def upgrade():
    # Define columns that need to be added/deleted
    user_columns = [Column('gid', String),
                    Column('display_name', String)]
    burst_columns = [Column('range1', String), Column('range2', String), Column('fk_simulation', Integer),
                     Column('fk_operation_group', Integer), Column('fk_metric_operation_group', Integer)]
    op_column = Column('view_model_disk_size', Integer)

    # Get tables
    inspector = Inspector.from_engine(conn)
    table_names = inspector.get_table_names()
    tables = Base.metadata.tables

    try:
        op.rename_table('BURST_CONFIGURATIONS', 'BurstConfiguration')

        # Dropping tables which don't exist in the new version
        op.drop_table('MAPPED_LOOK_UP_TABLE_DATA')
        op.drop_table('MAPPED_DATATYPE_MEASURE_DATA')
        op.drop_table('MAPPED_SPATIAL_PATTERN_VOLUME_DATA')
        op.drop_table('MAPPED_SIMULATION_STATE_DATA')
        op.drop_table('WORKFLOW_STEPS')
        op.drop_table('WORKFLOW_VIEW_STEPS')

        # Dropping tables which will be repopulated from the H5 files
        op.drop_table('MAPPED_COHERENCE_SPECTRUM_DATA')
        op.drop_table('MAPPED_COMPLEX_COHERENCE_SPECTRUM_DATA')
        op.drop_table('MAPPED_CONNECTIVITY_ANNOTATIONS_DATA')
        op.drop_table('MAPPED_CONNECTIVITY_MEASURE_DATA')
        op.drop_table('MAPPED_CONNECTIVITY_DATA')
        op.drop_table('MAPPED_CORRELATION_COEFFICIENTS_DATA')
        op.drop_table('MAPPED_COVARIANCE_DATA')
        op.drop_table('MAPPED_CROSS_CORRELATION_DATA')
        op.drop_table('MAPPED_FCD_DATA')
        op.drop_table('MAPPED_FOURIER_SPECTRUM_DATA')
        op.drop_table('MAPPED_INDEPENDENT_COMPONENTS_DATA')
        op.drop_table('MAPPED_LOCAL_CONNECTIVITY_DATA')
        op.drop_table('MAPPED_PRINCIPAL_COMPONENTS_DATA')
        op.drop_table('MAPPED_PROJECTION_MATRIX_DATA')
        op.drop_table('MAPPED_REGION_MAPPING_DATA')
        op.drop_table('MAPPED_REGION_VOLUME_MAPPING_DATA')
        op.drop_table('MAPPED_TIME_SERIES_REGION_DATA')
        op.drop_table('MAPPED_TIME_SERIES_EEG_DATA')
        op.drop_table('MAPPED_TIME_SERIES_MEG_DATA')
        op.drop_table('MAPPED_TIME_SERIES_SEEG_DATA')
        op.drop_table('MAPPED_TIME_SERIES_SURFACE_DATA')
        op.drop_table('MAPPED_TIME_SERIES_VOLUME_DATA')
        op.drop_table('MAPPED_TIME_SERIES_DATA')
        op.drop_table('MAPPED_SENSORS_DATA')
        op.drop_table('MAPPED_TRACTS_DATA')
        op.drop_table('MAPPED_STIMULI_REGION_DATA')
        op.drop_table('MAPPED_STIMULI_SURFACE_DATA')
        op.drop_table('MAPPED_STRUCTURAL_MRI_DATA')
        op.drop_table('MAPPED_SURFACE_DATA')
        op.drop_table('MAPPED_VALUE_WRAPPER_DATA')
        op.drop_table('MAPPED_VOLUME_DATA')
        op.drop_table('MAPPED_WAVELET_COEFFICIENTS_DATA')
        op.drop_table('DATA_TYPES_GROUPS')
        op.drop_table('MAPPED_ARRAY_DATA')
        op.drop_table('MAPPED_SPATIO_TEMPORAL_PATTERN_DATA')
        op.drop_table('MAPPED_SPATIAL_PATTERN_DATA')
        op.drop_table('WORKFLOWS')

        # Delete migrate_version if exists
        if 'migrate_version' in table_names:
            op.drop_table('migrate_version')
    except Exception as excep:
        LOGGER.exception(excep)

    # Migrating USERS table
    if TvbProfile.current.db.SELECTED_DB == 'postgres':
        op.add_column('USERS', user_columns[0])
        op.add_column('USERS', user_columns[1])
        op.create_unique_constraint('USERS_gid_key', 'USERS', ['gid'])
    else:
        with op.batch_alter_table('USERS', table_args=(UniqueConstraint('gid'),)) as batch_op:
            batch_op.add_column(user_columns[0])
            batch_op.add_column(user_columns[1])

    users_table = tables['USERS']
    user_ids = conn.execute("""SELECT U.id FROM "USERS" U""").fetchall()
    for id in user_ids:
        conn.execute(users_table.update().where(users_table.c.id == id[0]).
                     values({"gid": uuid.uuid4().hex, "display_name": users_table.c.username}))
    conn.execute('COMMIT')

    # Migrating BurstConfiguration table
    burst_config_table = tables['BurstConfiguration']
    for column in burst_columns:
        op.add_column('BurstConfiguration', column)

    try:
        op.alter_column('BurstConfiguration', '_dynamic_ids', new_column_name='dynamic_ids')
        op.alter_column('BurstConfiguration', '_simulator_configuration', new_column_name='simulator_gid')
        conn.execute(burst_config_table.delete().where(burst_config_table.c.status == 'error'))

        ranges = conn.execute("""SELECT OG.id, OG.range1, OG.range2 from "OPERATION_GROUPS" OG """).fetchall()

        ranges_1 = []
        ranges_2 = []

        for r in ranges:
            ranges_1.append(str(r[1]))
            ranges_2.append(str(r[2]))

        new_ranges_1 = _migrate_range_params(ranges_1)
        new_ranges_2 = _migrate_range_params(ranges_2)

        # Migrating Operation Groups
        operation_groups_table = tables['OPERATION_GROUPS']
        operation_groups = conn.execute("""SELECT * FROM "OPERATION_GROUPS" """).fetchall()

        for op_g in operation_groups:
            operation = conn.execute("""SELECT fk_operation_group, parameters, meta_data FROM "OPERATIONS" O """
                                     """WHERE O.fk_operation_group = """ + str(op_g[0])).fetchone()
            burst_id = eval(operation[2])['Burst_Reference']

            # Find if operation refers to an operation group or a metric operation group
            if 'time_series' in operation[1]:
                conn.execute(burst_config_table.update().where(burst_config_table.c.id == burst_id).
                             values({"fk_metric_operation_group": operation[0]}))
            else:
                conn.execute(burst_config_table.update().where(burst_config_table.c.id == burst_id).
                             values({"fk_operation_group": operation[0]}))

        for i in range(len(ranges_1)):
            range1 = str(new_ranges_1[i]).replace('\'', '')
            range2 = str(new_ranges_2[i]).replace('\'', '')
            _update_range_parameters(burst_config_table, operation_groups_table, range1, range2, ranges[i][0])

        conn.execute('COMMIT')
    except Exception as excep:
        LOGGER.exception(excep)

    # Finish BurstConfiguration migration by deleting unused column and adding foreign keys
    with op.batch_alter_table('BurstConfiguration') as batch_op:
        batch_op.drop_column('workflows_number')
        batch_op.create_foreign_key('bc_fk_simulation', 'OPERATIONS', ['fk_simulation'], ['id'])
        batch_op.create_foreign_key('bc_fk_operation_group', 'OPERATION_GROUPS', ['fk_operation_group'], ['id'])
        batch_op.create_foreign_key('bc_metric_operation_group', 'OPERATION_GROUPS',
                                    ['fk_metric_operation_group'],['id'])
    conn.execute('COMMIT')

    # MIGRATING Operations
    op_table = tables['OPERATIONS']
    try:
        burst_ref_metadata = conn.execute("""SELECT id, meta_data FROM "OPERATIONS" """
                                          """WHERE meta_data like '%%Burst_Reference%%' """).fetchall()
        op.alter_column('OPERATIONS', 'parameters', new_column_name='view_model_gid')

        for metadata in burst_ref_metadata:
            metadata_dict = eval(str(metadata[1]))
            conn.execute(op_table.update().where(op_table.c.id == metadata[0]).
                         values({'view_model_gid': json.dumps(metadata_dict['Burst_Reference'])}))

        op.rename_table('BurstConfiguration', 'BURST_CONFIGURATION')
        conn.execute('COMMIT')
    except Exception as excep:
        LOGGER.exception(excep)

    with op.batch_alter_table('OPERATIONS') as batch_op:
        batch_op.add_column(op_column)
        batch_op.drop_column('meta_data')
    conn.execute('COMMIT')

    try:
        op.drop_table('ALGORITHMS')
        op.drop_table('ALGORITHM_CATEGORIES')
        op.drop_table('DATA_TYPES')
    except Exception as excep:
        try:
            conn.execute("""DROP TABLE if exists "ALGORITHMS" cascade; """)
            conn.execute("""DROP TABLE if exists "ALGORITHM_CATEGORIES" cascade; """)
            conn.execute("""DROP TABLE if exists "DATA_TYPES" cascade; """)
        except Exception as excep:
            LOGGER.exception(excep)
        LOGGER.exception(excep)


def downgrade():
    """
    Downgrade currently not supported
    """
    pass
