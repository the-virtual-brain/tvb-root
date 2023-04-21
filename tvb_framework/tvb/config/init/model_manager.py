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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@invalid.tvb>
"""
import os
import shutil

import tvb.config.init.alembic.versions as scripts
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.sql import text
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.entities.storage import SA_SESSIONMAKER
from tvb.core.neotraits.db import Base

LOGGER = get_logger(__name__)


def initialize_startup():
    """ Force DB tables create, in case no data is already found."""
    is_db_empty = False
    session = SA_SESSIONMAKER()
    inspector = inspect(session.connection())
    table_names = inspector.get_table_names()
    if len(table_names) < 1:
        LOGGER.debug("Database access exception, maybe DB is empty")
        is_db_empty = True
    session.close()

    versions_repo = TvbProfile.current.db.DB_VERSIONING_REPO
    alembic_cfg = Config()
    alembic_cfg.set_main_option('script_location', versions_repo)
    alembic_cfg.set_main_option('sqlalchemy.url', TvbProfile.current.db.DB_URL)

    if is_db_empty:
        LOGGER.info("Initializing Database")
        if os.path.exists(versions_repo):
            shutil.rmtree(versions_repo)

        _update_sql_scripts()
        session = SA_SESSIONMAKER()
        Base.metadata.create_all(bind=session.connection())
        session.commit()
        session.close()

        command.stamp(alembic_cfg, 'head')
        LOGGER.info("Database Default Tables created successfully!")
    else:
        _update_sql_scripts()

        if 'migrate_version' in table_names:
            db_version = session.execute(text("""SELECT version from migrate_version""")).fetchone()[0]

            if db_version == 18:
                command.stamp(alembic_cfg, 'head')
                session.execute(text("""DROP TABLE "migrate_version";"""))
                session.commit()

                return is_db_empty

        if 'alembic_version' in table_names:
            db_version = session.execute(text("""SELECT version_num from alembic_version""")).fetchone()
            if not db_version:
                command.stamp(alembic_cfg, 'head')

        with session.connection() as connection:
            alembic_cfg.attributes['connection'] = connection
            command.upgrade(alembic_cfg, TvbProfile.current.version.DB_STRUCTURE_VERSION)
        LOGGER.info("Database already has some data, will not be re-created!")

        # Alembic throws an error if we use alembic_cfg with a connection that has been already closed
        del alembic_cfg.attributes['connection']
        command.stamp(alembic_cfg, 'head')

    return is_db_empty


def reset_database():
    """
    Remove all tables in DB.
    """
    LOGGER.warning("Your Database tables will be deleted.")
    try:
        session = SA_SESSIONMAKER()
        LOGGER.debug("Delete connection initiated.")
        inspector = inspect(session.connection())
        for table in inspector.get_table_names():
            try:
                LOGGER.debug("Removing:" + table)
                session.execute(text("DROP TABLE \"%s\" CASCADE" % table))
            except Exception:
                try:
                    session.execute(text("DROP TABLE %s" % table))
                except Exception as excep1:
                    LOGGER.error("Could no drop table %s", table)
                    LOGGER.exception(excep1)

        session.commit()
        LOGGER.info("Database was cleanup!")
    except Exception as excep:
        LOGGER.warning(excep)
    finally:
        session.close()


def _update_sql_scripts():
    """
    When a new release is done, make sure old DB scripts are updated.
    """
    scripts_folder = os.path.dirname(scripts.__file__)
    versions_folder = os.path.join(TvbProfile.current.db.DB_VERSIONING_REPO, 'versions')
    if os.path.exists(versions_folder):
        shutil.rmtree(versions_folder)
    ignore_patters = shutil.ignore_patterns('.svn')
    shutil.copytree(scripts_folder, versions_folder, ignore=ignore_patters)
    shutil.copyfile(os.path.join(os.path.dirname(__file__), 'alembic/env.py'),
                    os.path.join(versions_folder, os.pardir, 'env.py'))
