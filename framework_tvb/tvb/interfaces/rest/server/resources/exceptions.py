from abc import abstractmethod

from tvb.basic.exceptions import TVBException


class BaseRestException(TVBException):
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


class BadRequestException(BaseRestException):
    def __init__(self, message, payload=None):
        super().__init__(message, code=400, payload=payload)

    def get_default_message(self):
        return "Bad request error"
