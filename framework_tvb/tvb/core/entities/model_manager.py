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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@invalid.tvb>
"""
import os
import shutil
import migrate.versioning.api as migratesqlapi
from sqlalchemy.sql import text
from sqlalchemy.engine import reflection
from tvb.basic.config.settings import TVBSettings as cfg
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER
from tvb.basic.logger.builder import get_logger
import tvb.core.entities.model.db_update_scripts as scripts


LOGGER = get_logger(__name__)



def initialize_startup():
    """ Force DB tables create, in case no data is already found."""
    is_db_empty = False
    session = SA_SESSIONMAKER()
    inspector = reflection.Inspector.from_engine(session.connection())
    if len(inspector.get_table_names()) < 1:
        LOGGER.debug("Database access exception, maybe DB is empty")
        is_db_empty = True
    session.close()

    if is_db_empty:
        LOGGER.info("Initializing Database")
        if os.path.exists(cfg.DB_VERSIONING_REPO):
            shutil.rmtree(cfg.DB_VERSIONING_REPO)
        migratesqlapi.create(cfg.DB_VERSIONING_REPO, os.path.split(cfg.DB_VERSIONING_REPO)[1])
        _update_sql_scripts()
        migratesqlapi.version_control(cfg.DB_URL, cfg.DB_VERSIONING_REPO, version=cfg.DB_CURRENT_VERSION)
        session = SA_SESSIONMAKER()
        model.Base.metadata.create_all(bind=session.connection())
        session.commit()
        session.close()
        LOGGER.info("Database Default Tables created successfully!")
    else:
        _update_sql_scripts()
        migratesqlapi.upgrade(cfg.DB_URL, cfg.DB_VERSIONING_REPO, version=cfg.DB_CURRENT_VERSION)
        LOGGER.info("Database already has some data, will not be re-created!")
    return is_db_empty



def reset_database():
    """
    Remove all tables in DB.
    """
    LOGGER.warning("Your Database tables will be deleted.")
    try:
        session = SA_SESSIONMAKER()
        LOGGER.debug("Delete connection initiated.")
        inspector = reflection.Inspector.from_engine(session.connection())
        for table in inspector.get_table_names():
            try:
                LOGGER.debug("Removing:" + table)
                session.execute(text("DROP TABLE \"%s\" CASCADE" % table))
            except Exception:
                try:
                    session.execute(text("DROP TABLE %s" % table))
                except Exception, excep1:
                    LOGGER.error("Could no drop table %s", table)
                    LOGGER.exception(excep1)
        session.commit()
        LOGGER.info("Database was cleanup!")
    except Exception, excep:
        LOGGER.warning(excep)
    finally:
        session.close()



def _update_sql_scripts():
    """
    When a new release is done, make sure old DB scripts are updated.
    """
    scripts_folder = os.path.dirname(scripts.__file__)
    versions_folder = os.path.join(cfg.DB_VERSIONING_REPO, 'versions')
    if os.path.exists(versions_folder):
        shutil.rmtree(versions_folder)
    ignore_patters = shutil.ignore_patterns('.svn')
    shutil.copytree(scripts_folder, versions_folder, ignore=ignore_patters)

