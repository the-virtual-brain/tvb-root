# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
# Logs and errors

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.simulator.plot.config import OutputConfig


def initialize_logger(name, target_folder=OutputConfig().FOLDER_LOGS):
    """
    create logger for a given module. If the LIBRARY profile is not set, use the default TVB logger
    :param name: Logger Base Name
    :param target_folder: Folder where log files will be written
    """
    if not TvbProfile.is_library_mode():
        return get_logger(name)
    if not (os.path.isdir(target_folder)):
        os.makedirs(target_folder)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)

    fh = TimedRotatingFileHandler(os.path.join(target_folder, 'logs.log'), when="d", interval=1, backupCount=2)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # Log errors separately, to have them easy to inspect
    fhe = TimedRotatingFileHandler(os.path.join(target_folder, 'log_errors.log'), when="d", interval=1, backupCount=2)
    fhe.setFormatter(formatter)
    fhe.setLevel(logging.ERROR)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fhe)

    return logger


def raise_value_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nValueError: " + msg + "\n")
    raise ValueError(msg)


def raise_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nError: " + msg + "\n")
    raise Exception(msg)


def raise_import_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nImportError: " + msg + "\n")
    raise ImportError(msg)


def raise_not_implemented_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nNotImplementedError: " + msg + "\n")
    raise NotImplementedError(msg)


def warning(msg, logger=None):
    if logger is not None:
        logger.warning("\n" + msg + "\n")
