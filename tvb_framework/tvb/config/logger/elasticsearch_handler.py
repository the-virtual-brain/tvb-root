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
from logging.handlers import QueueListener, QueueHandler
from queue import Queue
from tvb.basic.profile import TvbProfile

if not TvbProfile.current.TRACE_USER_ACTIONS:

    class ElasticQueueHandler(Handler):
        def emit(self, record: LogRecord) -> None:
            pass

else:

    from elasticsearch import Elasticsearch


    def _convert_to_bulk_format(record):
        return [{"index": {}},
                {"@timestamp": record.asctime, "message": record.message, "user": {"id": "user-id"}}]


    class ElasticSendHandler(Handler):
        def __init__(self):
            """
            Initializes the custom http handler
            """
            super().__init__()
            self._client = Elasticsearch(
                TvbProfile.current.ELASTICSEARCH_URL,
                api_key=TvbProfile.current.ELASTICSEARCH_API_KEY,
                request_timeout=TvbProfile.current.ELASTICSEARCH_REQUEST_TIMEOUT
            )
            self.threshold = TvbProfile.current.ELASTICSEARCH_BUFFER_THRESHOLD
            self.buffer = []

        def emit(self, record: LogRecord):
            """
            This function gets called when a log event gets emitted. It recieves a
            record, formats it and sends it to the url
            Parameters:
                record: a log record
            """
            self.buffer += _convert_to_bulk_format(record)

            if len(self.buffer) // 2 >= self.threshold:
                self._client.bulk(index=TvbProfile.current.ELASTICSEARCH_LOGGING_INDEX, operations=self.buffer)
                self.buffer.clear()

        def close(self) -> None:
            self._client.close()
            self.buffer = []
            return super().close()


    class ElasticQueueHandler(QueueHandler):

        def __init__(self):
            # sets the queue attribute
            super().__init__(Queue(-1))
            self.sending_handler = ElasticSendHandler()
            self._listener = QueueListener(self.queue, self.sending_handler)
            self._listener.start()

        def close(self) -> None:
            self._listener.stop()
            self.sending_handler.close()
            return super().close()
