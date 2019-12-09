import datetime
import json
from json import JSONEncoder
from uuid import UUID

import numpy
from sqlalchemy.orm.state import InstanceState
from tvb.basic.logger.builder import get_logger


class DatatypeEncoder(JSONEncoder):

    def default(self, obj):
        Logger = get_logger(obj.__class__.__module__)

        if isinstance(obj, UUID):
            return {'gid': obj.hex}
        elif isinstance(obj, type(Logger)):
            return {'logger': obj.name}
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AlgorithmEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (InstanceState, datetime.datetime)):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data)  # this will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)

