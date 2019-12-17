from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.server.dto.dtos import UserDto

from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class GetUsersResource(RestResource):
    def get(self):
        users, _ = UserService.retrieve_all_users('dummy')
        return [UserDto(user) for user in users]
