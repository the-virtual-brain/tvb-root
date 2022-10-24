# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import os
import h5py


class Base(object):

    h5_path = ""

    @property
    def _hdf_file(self):
        return None

    def _set_hdf_file(self, hfile):
        pass

    @property
    def _fmode(self):
        return None

    @property
    def _mode(self):
        return ""

    @property
    def _mode_past(self):
        return ""

    @property
    def _to_from(self):
        return ""

    def _open_file(self, type_name=""):
        try:
            self._set_hdf_file(h5py.File(self.h5_path, self._fmode, libver='latest'))
        except Exception as e:
            self.logger.warning("Could not open file %s\n%s!" % (self.h5_path, str(e)))

    def _assert_file(self, path=None, type_name=""):
        if path is not None:
            self.h5_path = path
        if self._hdf_file is not None:
            self._close_file()
        self._open_file(type_name)
        self.logger.info("Starting to %s %s %s H5 file: %s" % (self._mode, type_name, self._to_from, self.h5_path))

    def _close_file(self, close_file=True):
        if close_file:
            try:
                self._hdf_file.close()
                self._set_hdf_file(None)
            except Exception as e:
                self.logger.warning("Could not close file %s\n%s!" % (self.h5_path, str(e)))

    def _log_success_or_warn(self, exception=None, type_name=""):
        if exception is None:
            self.logger.info("Successfully %s %s %s H5 file: %s" %
                             (self._mode_past, type_name, self._to_from, self.h5_path))
        else:
            self.logger.warning("Failed to %s %s %s H5 file %s\n%s!" %
                                (self._mode, type_name, self._to_from, self.h5_path, str(exception)))
