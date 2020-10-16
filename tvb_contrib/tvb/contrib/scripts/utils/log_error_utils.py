# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

"""
Logs and errors

.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""
import time


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


def print_toc_message(tic):
    toc = time.time() - tic
    if toc > 60.0:
        if toc > 3600.0:
            toc /= 3600.0
            unit = "hours"
        else:
            toc /= 60.0
            unit = "mins"
    else:
        unit = "sec"
    print("DONE in %f %s!" % (toc, unit))
