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

import uuid
from tvb.basic.neotraits.api import Attr
from tvb.core.neotraits.h5 import H5File, Scalar, Reference
from tvb.core.utils import string2date, date2string
from tvb.storage.h5.file.exceptions import MissingDataSetException


class BurstConfigurationH5(H5File):
    def __init__(self, path):
        super(BurstConfigurationH5, self).__init__(path)
        self.name = Scalar(Attr(str), self, name='name')
        self.status = Scalar(Attr(str), self, name='status')
        self.error_message = Scalar(Attr(str, required=False), self, name='error_message')
        self.start_time = Scalar(Attr(str), self, name='start_time')
        self.finish_time = Scalar(Attr(str, required=False), self, name='finish_time')
        self.simulator = Reference(Attr(uuid.UUID), self, name='simulator')
        self.range1 = Scalar(Attr(str, required=False), self, name='range1')
        self.range2 = Scalar(Attr(str, required=False), self, name='range2')

    def store(self, burst_config, scalars_only=False, store_references=True):
        # type (BurstConfiguration, bool, bool) -> None
        self.gid.store(uuid.UUID(burst_config.gid))
        self.name.store(burst_config.name)
        self.status.store(burst_config.status)
        self.error_message.store(burst_config.error_message or 'None')
        self.start_time.store(date2string(burst_config.start_time))
        self.finish_time.store(date2string(burst_config.finish_time))
        self.simulator.store(uuid.UUID(burst_config.simulator_gid))
        self.range1.store(burst_config.range1)
        self.range2.store(burst_config.range2)

    def load_into(self, burst_config):
        # type (BurstConfiguration) -> None
        burst_config.gid = self.gid.load().hex
        burst_config.name = self.name.load()
        burst_config.status = self.status.load()
        burst_config.error_message = self.error_message.load()
        burst_config.start_time = string2date(self.start_time.load())
        finish_time = self.finish_time.load()
        if finish_time and finish_time != 'None':
            burst_config.finish_time = string2date(finish_time)
        burst_config.simulator_gid = self.simulator.load().hex
        try:
            burst_config.range1 = self.range1.load()
        except MissingDataSetException:
            burst_config.range1 = None
        try:
            burst_config.range2 = self.range2.load()
        except MissingDataSetException:
            burst_config.range2 = None
