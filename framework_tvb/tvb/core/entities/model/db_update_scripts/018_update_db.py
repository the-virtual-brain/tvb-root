# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Change of DB structure to TVB 2.0

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
from migrate import alter_column, create_column, drop_column
from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey
from sqlalchemy.engine import reflection
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import Operation, OperationGroup
from tvb.core.entities.storage import SA_SESSIONMAKER
from sqlalchemy.sql import text
from tvb.core.neotraits.db import Base


meta = Base.metadata

LOGGER = get_logger(__name__)

COLUMN_1_OLD = Column('_dynamic_ids', String)
COLUMN_1_NEW = Column('dynamic_ids', String, nullable=False)
COLUMN_2 = Column('range_1', String)
COLUMN_3 = Column('range_2', String)
COLUMN_4_OLD = Column('_simulator_configuration', String)
COLUMN_4_NEW = Column('simulator_gid', String)
COLUMN_5 = Column('fk_simulation', Integer, ForeignKey(Operation.id))
COLUMN_6 = Column('fk_operation_group', Integer, ForeignKey(OperationGroup.id))
COLUMN_7 = Column('fk_metric_operation_group', Integer, ForeignKey(Operation.id))

COLUMN_8_OLD = Column('_source', String)
COLUMN_8_NEW = Column('fk_source_gid', String(32), nullable=False)
COLUMN_9_OLD = Column('_nfft', Integer)
COLUMN_9_NEW = Column('nfft', Integer, nullable=False)
COLUMN_10 = Column('_frequency', String)
COLUMN_11 = Column('frequencies_min', Float)
COLUMN_12 = Column('frequencies_max', Float)
COLUMN_13 = Column('_cross_spectrum', String)
COLUMN_14_OLD = Column('_epoch_length', Float)
COLUMN_14_NEW = Column('epoch_length', Float, nullable=False)
COLUMN_15_OLD = Column('_segment_length', Float)
COLUMN_15_NEW = Column('segment_length', Float, nullable=False)
COLUMN_16_OLD = Column('_windowing_function', String)
COLUMN_16_NEW = Column('windowing_function', String, nullable=False)
COLUMN_17 = Column('frequency_step', Float, nullable=False)
COLUMN_18 = Column('max_frequency', Float, nullable=False)


COLUMN_19_OLD = Column('_connectivity', String)
COLUMN_19_NEW = Column('fk_connectivity_gid', String(32))
COLUMN_20 = Column('_region_annotations', String)
COLUMN_21 = Column('annotations_length', Integer)

COLUMN_22_OLD = Column('_number_of_regions', Integer)
COLUMN_22_NEW = Column('number_of_regions', Integer, nullable=False)
COLUMN_23_OLD = Column('_number_of_connections', Integer)
COLUMN_23_NEW = Column('number_of_connections', Integer, nullable=False)
COLUMN_24_OLD = Column('_undirected', Integer)
COLUMN_24_NEW = Column('undirected', Boolean)
COLUMN_25 = Column('_weights', String)
COLUMN_26 = Column('weights_min', Float)
COLUMN_27 = Column('weights_max', Float)
COLUMN_28 = Column('weights_mean', Float)
COLUMN_29 = Column('_tract_lengths', Float)
COLUMN_30 = Column('tract_lengths_min', Float)
COLUMN_31 = Column('tract_lengths_max', Float)
COLUMN_32 = Column('tract_lengths_mean', Float)
COLUMN_33 = Column('has_cortical_mask', Float)
COLUMN_34 = Column('has_hemispheres_mask', Float)
COLUMN_35 = Column('_cortical', String)
COLUMN_36 = Column('_delays', String)
COLUMN_37 = Column('_centres', String)
COLUMN_38 = Column('_idelays', String)
COLUMN_39 = Column('_hemispheres', String)
COLUMN_40 = Column('_orientations', String)
COLUMN_41 = Column('_region_labels', String)
COLUMN_42 = Column('_saved_selection', String)
COLUMN_43 = Column('_speed', String)
COLUMN_44 = Column('_parent_connectivity', String)
COLUMN_45 = Column('_areas', String)

COLUMN_46 = Column('has_surface_mapping', Boolean, nullable=False)


def upgrade(migrate_engine):
    """
    """
    meta.bind = migrate_engine

    session = SA_SESSIONMAKER()
    inspector = reflection.Inspector.from_engine(session.connection())

    try:
        session.execute(text("""DROP TABLE "MAPPED_ARRAY_DATA";"""))
        session.execute(text("""DROP TABLE "DATA_TYPES";"""))
        session.execute(text("""DROP TABLE "MAPPED_LOOK_UP_TABLE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_DATATYPE_MEASURE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SPATIAL_PATTERN_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SPATIAL_PATTERN_VOLUME_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SIMULATION_STATE_DATA";"""))
        session.execute(text("""DROP TABLE "WORKFLOWS";"""))
        session.execute(text("""DROP TABLE "WORKFLOW_STEPS";"""))
        session.execute(text("""DROP TABLE "WORKFLOW_VIEW_STEPS";"""))

        for table in inspector.get_table_names():
            new_table_name = table
            new_table_name = new_table_name.lower()
            if 'mapped' in new_table_name:
                new_table_name = list(new_table_name)
                for i in range(len(new_table_name)):
                    if new_table_name[i] == '_':
                        new_table_name[i + 1] = new_table_name[i + 1].upper()

                new_table_name = "".join(new_table_name)
                new_table_name = new_table_name.replace('_', '')
                new_table_name = new_table_name.replace('mapped', '')
                new_table_name = new_table_name.replace('Data', '')
                new_table_name = new_table_name + 'Index'

                session.execute(text("""ALTER TABLE "{}" RENAME TO "{}"; """.format(table, new_table_name)))

        session.execute(text("""ALTER TABLE "BURST_CONFIGURATIONS"
                                RENAME TO "BurstConfiguration"; """))
        session.execute(text("""ALTER TABLE "StructuralMriIndex"
                                RENAME TO "StructuralMRIIndex"; """))
        session.execute(text("""ALTER TABLE "TimeSeriesEegIndex"
                                RENAME TO "TimeSeriesEEGIndex"; """))
        session.execute(text("""ALTER TABLE "TimeSeriesMegIndex"
                                RENAME TO "TimeSeriesMEGIndex"; """))
        session.execute(text("""ALTER TABLE "TimeSeriesSeegIndex"
                                RENAME TO "TimeSeriesSEEGIndex"; """))

        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    burst_config_table = meta.tables['BurstConfiguration']
    alter_column(COLUMN_1_OLD, table=burst_config_table, name=COLUMN_1_NEW.name)
    create_column(COLUMN_2, burst_config_table)
    create_column(COLUMN_3, burst_config_table)
    alter_column(COLUMN_4_OLD, table=burst_config_table, name=COLUMN_4_NEW.name)
    create_column(COLUMN_5, burst_config_table)
    create_column(COLUMN_6, burst_config_table)
    create_column(COLUMN_7, burst_config_table)

    # session = SA_SESSIONMAKER()
    # try:
    #     session.execute(text("""UPDATE "BurstConfiguration" SET fk_simulation =
    #      (SELECT )"""))

    coherence_spectrum_table = meta.tables['CoherenceSpectrumIndex']
    alter_column(COLUMN_8_OLD, table=coherence_spectrum_table, name=COLUMN_8_NEW.name)
    alter_column(COLUMN_9_OLD, table=coherence_spectrum_table, name=COLUMN_9_NEW.name)
    drop_column(COLUMN_10, table=coherence_spectrum_table)
    create_column(COLUMN_11, table=coherence_spectrum_table)
    create_column(COLUMN_12, table=coherence_spectrum_table)

    complex_coherence_spectrum_table = meta.tables['ComplexCoherenceSpectrumIndex']
    alter_column(COLUMN_8_OLD, table=complex_coherence_spectrum_table, name=COLUMN_8_NEW.name)
    drop_column(COLUMN_13, table=complex_coherence_spectrum_table)
    alter_column(COLUMN_14_OLD, table=complex_coherence_spectrum_table, name=COLUMN_14_NEW.name)
    alter_column(COLUMN_15_OLD, table=complex_coherence_spectrum_table, name=COLUMN_15_NEW.name)
    alter_column(COLUMN_16_OLD, table=complex_coherence_spectrum_table, name=COLUMN_16_NEW.name)
    create_column(COLUMN_17, table=complex_coherence_spectrum_table)
    create_column(COLUMN_18, table=complex_coherence_spectrum_table)

    connectivity_annotations_table = meta.tables['ConnectivityAnnotationsIndex']
    alter_column(COLUMN_19_OLD, table=connectivity_annotations_table, name=COLUMN_19_NEW.name)
    drop_column(COLUMN_20, table=connectivity_annotations_table)
    create_column(COLUMN_21, table=connectivity_annotations_table)

    connectivity_table = meta.tables['ConnectivityIndex']
    alter_column(COLUMN_22_OLD, table=connectivity_table, name=COLUMN_22_NEW.name)
    alter_column(COLUMN_23_OLD, table=connectivity_table, name=COLUMN_23_NEW.name)
    alter_column(COLUMN_24_OLD, table=connectivity_table, name=COLUMN_24_NEW.name)
    drop_column(COLUMN_25, table=connectivity_table)
    create_column(COLUMN_26, table=connectivity_table)
    create_column(COLUMN_27, table=connectivity_table)
    create_column(COLUMN_28, table=connectivity_table)
    drop_column(COLUMN_29, table=connectivity_table)
    create_column(COLUMN_30, table=connectivity_table)
    create_column(COLUMN_31, table=connectivity_table)
    create_column(COLUMN_32, table=connectivity_table)
    create_column(COLUMN_33, table=connectivity_table)
    create_column(COLUMN_34, table=connectivity_table)
    drop_column(COLUMN_35, table=connectivity_table)
    drop_column(COLUMN_36, table=connectivity_table)
    drop_column(COLUMN_37, table=connectivity_table)
    drop_column(COLUMN_38, table=connectivity_table)
    drop_column(COLUMN_39, table=connectivity_table)
    drop_column(COLUMN_40, table=connectivity_table)
    drop_column(COLUMN_41, table=connectivity_table)
    drop_column(COLUMN_42, table=connectivity_table)
    drop_column(COLUMN_43, table=connectivity_table)
    drop_column(COLUMN_44, table=connectivity_table)
    drop_column(COLUMN_45, table=connectivity_table)

    connectivity_measure_table = meta.tables['ConnectivityMeasureIndex']
    alter_column(COLUMN_19_OLD, table=connectivity_measure_table, name=COLUMN_19_NEW)
    create_column(COLUMN_46, table=connectivity_measure_table)




def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
