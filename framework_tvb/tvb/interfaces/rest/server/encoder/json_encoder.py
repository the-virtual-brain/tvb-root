import json
from json import JSONEncoder
from logging import Logger
from uuid import UUID

import numpy
from tvb.basic.logger.builder import get_logger


class CustomEncoder(JSONEncoder):

    def default(self, obj):
        Logger = get_logger(obj.__class__.__module__)

        if isinstance(obj, UUID):
            return {'gid': obj.hex}
        elif isinstance(obj, type(Logger)):
            return {'logger': obj.name}
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
