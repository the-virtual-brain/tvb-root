from flask import jsonify
from flask_restful import Resource
from tvb.core.services.user_service import UserService


class GetUsersResource(Resource):

    def get(self):
        users, _ = UserService.retrieve_all_users('dummy')
        final_dict = dict()
        for user in users:
            dict_user = dict(
                username=user.username,
                password=user.password,
                email=user.email,
                validated=user.validated,
                role=user.role
            )

            final_dict[user.username] = dict_user
        return jsonify({'users': final_dict})

    
