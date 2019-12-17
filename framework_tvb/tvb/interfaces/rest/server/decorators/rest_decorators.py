from functools import wraps

from flask import current_app
from flask.json import dumps


def _convert(obj):
    try:
        return obj.__dict__
    except AttributeError:
        return current_app.json_encoder().default(obj)


def rest_jsonify(func):
    @wraps(func)
    def deco(*a, **b):
        result = func(*a, **b)
        data = result
        status = 200
        if isinstance(result, tuple):
            data = result[0]
            status = result[1]
        if data is None:
            data = ''
        return current_app.response_class(dumps(data, default=lambda o: _convert(o), sort_keys=False),
                                          mimetype=current_app.config['JSONIFY_MIMETYPE'],
                                          status=status)

    return deco
