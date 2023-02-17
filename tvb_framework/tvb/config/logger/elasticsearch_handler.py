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
import logging
from logging import Handler, LogRecord
from logging.handlers import QueueListener, QueueHandler
from queue import Queue

from elasticsearch import Elasticsearch

from tvb.basic.profile import TvbProfile


class ElasticSendHandler(Handler):
    def __init__(self):
        """
                Initializes the custom http handler
        """
        super().__init__()
        self._client = Elasticsearch(
            "https://elk-cscs.tc.humanbrainproject.eu:9200",
            api_key='MzdnQVg0WUJFazFQZTBNTlBuZ3I6dFJuRmNLeThUQm12YmNOc0RLVFdsUQ==',
            request_timeout=30
        )
        self.threshold = 5
        self.buffer = []

    def _convert_to_bulk_format(self, record):
        return [{"index": {}}, {"@timestamp": record.asctime, "message": record.message, "user": {"id": "user-id"}}]

    def emit(self, record: LogRecord):
        """
        This function gets called when a log event gets emitted. It recieves a
        record, formats it and sends it to the url
        Parameters:
            record: a log record
        """

        self.buffer += self._convert_to_bulk_format(record)

        if len(self.buffer) // 2 >= self.threshold:
            self._client.bulk(
                index="app_tvb_logging",
                operations=self.buffer
            )
            self.buffer.clear()

    def close(self) -> None:
        self._client.close()
        self.buffer = []
        return super().close()


if TvbProfile.current.TRACE_USER_ACTIONS:
    class ElasticQueueHandler(QueueHandler):
        def __init__(self):
            # sets the queue attribute
            super().__init__(Queue(-1))

            self.sending_handler = ElasticSendHandler()
            self._listener = QueueListener(self.queue,
                                           self.sending_handler)
            self._listener.start()

        def close(self) -> None:
            self._listener.stop()
            self.sending_handler.close()
            return super().close()
else:  # if not TvbProfile.current.TRACE_USER_ACTIONS:
    class ElasticQueueHandler(Handler):
        def __init__(self):
            super.__init__()
        def emit(self, record: LogRecord) -> None:
            pass
