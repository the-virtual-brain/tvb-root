from abc import abstractmethod


class BaseRestException(Exception):
    def __init__(self, message=None, code=None, payload=None):
        Exception.__init__(self)
        self.message = message if message is not None else self.get_default_message()
        self.code = code
        self.payload = payload

    def to_dict(self):
        payload = dict(self.payload or ())
        payload['message'] = self.message
        payload['code'] = self.code
        return payload

    @abstractmethod
    def get_default_message(self):
        return None
