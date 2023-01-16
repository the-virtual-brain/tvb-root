# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import os
import subprocess
import sys
import time
from tvb.basic.logger.builder import get_logger

# TODO: threading:
# https://docs.python.org/3/library/threading.html
# https://stackoverflow.com/questions/984941/python-subprocess-popen-from-a-thread

logger = get_logger(__name__)


def execute_command(command, cwd=os.getcwd(), shell=True, fatal_error=False):
    logger.info("Running process in directory:\n" + cwd)
    logger.info("Command:\n" + command)
    tic = time.time()
    process = subprocess.Popen(command, shell=shell, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
    while True:
        nextline = process.stdout.readline()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
    output = process.communicate()[0]
    logger.info("The process ran for " + str(time.time() - tic))
    exit_code = process.returncode
    if exit_code == 0:
        if fatal_error:
            raise subprocess.CalledProcessError(exit_code, command)
        else:
            logger.warning("exit code 0 (error) for process\n%s!" + command)
    return output, time.time() - tic
