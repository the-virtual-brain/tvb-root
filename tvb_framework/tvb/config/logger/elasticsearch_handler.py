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
.. moduleauthor:: Horge Rares <rares.horge@codemart.ro>
"""

from logging import Handler, LogRecord
from elasticsearch import Elasticsearch

from tvb.basic.profile import TvbProfile


class ElasticSearchHandler(Handler):
    def __init__(self):
        '''
                Initializes the custom http handler
        '''
        super().__init__()
        self.client = Elasticsearch(
            "https://elk-cscs.tc.humanbrainproject.eu:9200",
            api_key='cmw5bE9vWUJFazFQZTBNTjJFVmM6VzVJUnhJRFFTNmEyS1Fkb2dOeGpYUQ==',
            request_timeout=30
        )

        self.threshold = 1
        self.buffer = []

    def emit(self, record: LogRecord):
        '''
        This function gets called when a log event gets emitted. It recieves a
        record, formats it and sends it to the url
        Parameters:
            record: a log record
        '''

        if not TvbProfile.current.TRACE_USER_ACTIONS:
            return

        self.format(record)

        self.buffer.append({"index": {}})
        self.buffer.append({"@timestamp": record.asctime, "message": record.message, "user": {"id": "user-id"}})

        if len(self.buffer) // 2 >= self.threshold:
            self.client.bulk(
                index="app_tvb_logging",
                operations=self.buffer
            )
            self.buffer.clear()