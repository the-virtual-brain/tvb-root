from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.server.dto.dtos import UserDto, ProjectDto

from tvb.interfaces.rest.server.resources.rest_resource import RestResource


class GetUsersResource(RestResource):
    """
    :return a list of system's users
    """

    def get(self):
        users = UserService.fetch_all_users()
        return [UserDto(user) for user in users]


class GetProjectsListResource(RestResource):
    """
    :return a list of user's projects
    """

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, user_id):
        projects, _ = self.project_service.retrieve_all_user_projects(user_id=user_id)
        return [ProjectDto(project) for project in projects]
