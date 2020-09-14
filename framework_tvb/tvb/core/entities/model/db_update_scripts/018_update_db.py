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
from migrate import create_column, drop_column, ForeignKeyConstraint
from sqlalchemy import Column, String, Integer, Float, Boolean, Date, Table
from sqlalchemy.engine import reflection
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage import SA_SESSIONMAKER, dao
from sqlalchemy.sql import text
from tvb.core.neotraits.db import Base, HasTraitsIndex
from tvb.core.neocom import h5

meta = Base.metadata

LOGGER = get_logger(__name__)


BURST_COLUMNS = [Column('range1', String), Column('range2', String), Column('fk_simulation', Integer),
Column('fk_operation_group', Integer), Column('fk_metric_operation_group', Integer)]
BURST_DELETED_COLUMN = Column('workflows_number', Integer)

CONN_COLUMNS = [Column('weights_min', Float), Column('weights_max', Float), Column('weights_mean', Float),
                Column('tract_lengths_min', Float), Column('tract_lengths_max', Float),
                Column('tract_lengths_mean', Float), Column('has_cortical_mask', Boolean),
                Column('has_hemispheres_mask', Boolean)]
CONN_DELETED_COLUMNS = [Column('_cortical', String), Column('_delays', String), Column('_centres', String),
                        Column('_idelays', String), Column('_hemispheres', String), Column('_orientations', String),
                        Column('_region_labels', String), Column('_saved_selection', String), Column('_speed', String),
                        Column('_parent_connectivity', String), Column('_areas', String)]

DATATYPE_DELETED_COLUMNS = [Column('gid', String), Column('create_date', Date)]


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
COLUMN_46 = Column('has_surface_mapping', Boolean, nullable=False)


def upgrade(migrate_engine):
    """
    """
    meta.bind = migrate_engine
    session = SA_SESSIONMAKER()

    # Renaming tables which wouldn't be correctly renamed by the next renamings
    try:
        session.execute(text("""ALTER TABLE "BURST_CONFIGURATIONS"
                                RENAME TO "BurstConfiguration"; """))
        session.execute(text("""ALTER TABLE "MAPPED_STRUCTURAL_MRI_DATA"
                                RENAME TO "StructuralMRIIndex"; """))
        session.execute(text("""ALTER TABLE "MAPPED_TIME_SERIES_EEG_DATA"
                                RENAME TO "TimeSeriesEEGIndex"; """))
        session.execute(text("""ALTER TABLE "MAPPED_TIME_SERIES_MEG_DATA"
                                RENAME TO "TimeSeriesMEGIndex"; """))
        session.execute(text("""ALTER TABLE "MAPPED_TIME_SERIES_SEEG_DATA"
                                RENAME TO "TimeSeriesSEEGIndex"; """))
        session.execute(text("""ALTER TABLE "DATA_TYPES"
                                RENAME TO "DataType"; """))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    session = SA_SESSIONMAKER()
    inspector = reflection.Inspector.from_engine(session.connection())

    try:
        # Dropping tables which don't exist in the new version

        session.execute(text("""DROP TABLE "MAPPED_ARRAY_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_LOOK_UP_TABLE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_DATATYPE_MEASURE_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SPATIAL_PATTERN_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SPATIAL_PATTERN_VOLUME_DATA";"""))
        session.execute(text("""DROP TABLE "MAPPED_SIMULATION_STATE_DATA";"""))
        session.execute(text("""DROP TABLE "WORKFLOWS";"""))
        session.execute(text("""DROP TABLE "WORKFLOW_STEPS";"""))
        session.execute(text("""DROP TABLE "WORKFLOW_VIEW_STEPS";"""))

        # This for renames the other tables replacing the "MAPPED_ ... _DATA" name structure with the "Ïndex" suffix
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

        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    # CREATING HasTraitsIndex
    # meta = MetaData(bind=migrate_engine)

    table = Table('HasTraitsIndex', meta, autoload=True)
    for constraint in table._sorted_constraints:
        if not isinstance(constraint.name, str):
            constraint.name = None

    meta.create_all()
    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""INSERT into "HasTraitsIndex" (id, gid, type_, title, create_date)
                            SELECT D.id, D.gid, D.type, NULL, D.create_date
                            FROM "Datatype" D"""))
        session.commit()
        session.execute(text("""UPDATE "HasTraitsIndex" 
                                    SET type_ =
                                     CASE
                                        WHEN HasTraitsIndex.type_ in ('CorticalSurface', 'BrainSkull', 'SkullSkin',
                                          'SkinAir', 'EEGCap', 'FaceSurface', 'WhiteMatterSurface') THEN 'SurfaceIndex'
                                        WHEN HasTraitsIndex.type_ = 'LocalConnectivity' THEN 'LocalConnectivityIndex'
                                        WHEN HasTraitsIndex.type_ in ('SensorsInternal', 'SensorsMEG', 'SensorsEEG') 
                                           THEN 'SensorsIndex'
                                        WHEN HasTraitsIndex.type_ in ('ProjectionSurfaceEEG', 'ProjectionSurfaceMEG',
                                         'ProjectionSurfaceSEEG') THEN 'ProjectionMatrixIndex'
                                        WHEN HasTraitsIndex.type_ = 'Connectivity' THEN 'ConnectivityIndex'
                                        WHEN HasTraitsIndex.type_ = 'RegionMapping' THEN 'RegionMappingIndex'
                                        WHEN HasTraitsIndex.type_ = 'ConnectivityAnnotations' THEN
                                         'ConnectivityAnnotationsIndex'
                                        WHEN HasTraitsIndex.type_ = 'TimeSeriesRegion' THEN 'TimeSeriesRegionIndex'
                                        WHEN HasTraitsIndex.type_ = 'ComplexCoherenceSpectrum'
                                         THEN 'ComplexCoherenceSpectrumIndex'
                                        WHEN HasTraitsIndex.type_ = 'CoherenceSpectrum' THEN 'CoherenceSpectrumIndex'
                                     END
                            """))
        session.execute(text("""DELETE FROM "HasTraitsIndex" WHERE type_ = 'SimulationState'"""))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    # MIGRATING Datatype
    datatype_table = meta.tables['DataType']
    for column in DATATYPE_DELETED_COLUMNS:
        drop_column(column, datatype_table)

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""UPDATE "DataType" SET type =
                        (SELECT HTI.type_
                        FROM "HasTraitsIndex" HTI, "DataType" DT
                        WHERE HTI.id = DT.id);"""))
        session.execute(text("""UPDATE "DataType" 
                                    SET module =
                                     CASE
                                        WHEN DataType.type = 'ConnectivityIndex' THEN 'tvb.adapters.datatypes.db.connectivity'
                                        WHEN DataType.type = 'SurfaceIndex' THEN 'tvb.adapters.datatypes.db.surface'
                                        WHEN DataType.type = 'SensorsIndex' THEN 'tvb.adapters.datatypes.db.sensors'
                                        WHEN DataType.type = 'ConnectivityAnnotationsIndex' THEN
                                        'tvb.adapters.datatypes.db.annotation'
                                        WHEN DataType.type = 'RegionMappingIndex' THEN 
                                        'tvb.adapters.datatypes.db.region_mapping'
                                        WHEN DataType.type  = 'ProjectionMatrixIndex' THEN 'tvb.adapters.datatypes.db.projections'
                                        WHEN DataType.type = 'LocalConnectivityIndex' THEN 'tvb.adapters.datatypes.db.local_connectivity'
                                        WHEN DataType.type = 'TimeSeriesRegionIndex' THEN 'tvb.adapters.datatypes.db.time_series'
                                        WHEN DataType.type in('ComplexCoherenceSpectrumIndex', 'CoherenceSpectrumIndex')  THEN
                                        'tvb.adapters.datatypes.db.spectral'
                                     END
                            """))
        session.commit()
    except:
        return False
    finally:
        session.close()

    burst_config_table = meta.tables['BurstConfiguration']
    for column in BURST_COLUMNS:
        create_column(column, burst_config_table)

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""ALTER TABLE "BurstConfiguration"
                                RENAME COLUMN _dynamic_ids TO dynamic_ids"""))
        session.execute(text("""ALTER TABLE "BurstConfiguration"
                                RENAME COLUMN _simulator_configuration TO simulator_gid"""))

        session.execute(text(
            """UPDATE "BurstConfiguration" SET
            fk_simulation = (SELECT O.id FROM "OPERATIONS" O, "DataType" D
             WHERE O.id = D.fk_from_operation AND module = 'tvb.adapters.datatypes.db.time_series')"""))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    # Drop old column
    drop_column(BURST_DELETED_COLUMN, burst_config_table)

    # Create constraints only after the rows are populated
    fk_burst_config_constraint_1 = ForeignKeyConstraint(
        ["fk_simulation"],
        ["OPERATIONS.id"],
        table=burst_config_table)
    fk_burst_config_constraint_2 = ForeignKeyConstraint(
        ["fk_operation_group"],
        ["OPERATION_GROUPS.id"],
        table=burst_config_table)
    fk_burst_config_constraint_3 = ForeignKeyConstraint(
        ["fk_metric_operation_group"],
        ["OPERATION_GROUPS.id"],
        table=burst_config_table)

    fk_burst_config_constraint_1.create()
    fk_burst_config_constraint_2.create()
    fk_burst_config_constraint_3.create()

    # MIGRATING ConnectivityIndex #

    conn_table = meta.tables['ConnectivityIndex']
    for column in CONN_COLUMNS:
        create_column(column, conn_table)

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("""ALTER TABLE "ConnectivityIndex"
                                RENAME COLUMN _number_of_regions TO number_of_regions"""))
        session.execute(text("""ALTER TABLE "ConnectivityIndex"
                                RENAME COLUMN _number_of_connections TO number_of_connections"""))
        session.execute(text("""ALTER TABLE "ConnectivityIndex"
                                RENAME COLUMN _undirected TO undirected"""))
        session.commit()
    except Exception:
        session.close()
    finally:
        session.close()

    for column in CONN_DELETED_COLUMNS:
        drop_column(column, conn_table)


def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
