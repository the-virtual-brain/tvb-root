from flask_restful import Api
from tvb.basic.exceptions import TVBException


class RestApi(Api):
    def handle_error(self, e):
        if not isinstance(e, TVBException):
            super().handle_error(e)

        code = getattr(e, 'code', 500)
        message = getattr(e, 'message', 'Internal Server Error')
        to_dict = getattr(e, 'to_dict', None)

        if to_dict:
            data = to_dict()
        else:
            data = {'message': message}
        return self.make_response(data, code)
