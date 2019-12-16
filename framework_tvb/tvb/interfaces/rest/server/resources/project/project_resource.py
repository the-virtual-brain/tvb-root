from flask import jsonify
from flask_restful import Resource
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.flow_service import FlowService
from tvb.core.services.project_service import ProjectService
import json
from tvb.interfaces.rest.server.encoders.json_encoders import CustomEncoder


class GetProjectsListResource(Resource):

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, user_id):
        projects, _ = self.project_service.retrieve_projects_for_user(user_id=user_id)
        final_dict = dict()
        for project in projects:
            dict_project = dict(
                id=project.id,
                name=project.name,
                description=project.description,
                last_updated=project.last_updated,
                fk_admin=project.fk_admin,
                gid=project.gid,
                version=project.version
            )

            final_dict[project.name] = dict_project
        return jsonify({'projects': final_dict})


class GetDataInProjectResource(Resource):

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, project_id):
        project = self.project_service.find_project(project_id)
        datatypes = self.project_service.get_project_structure(project,  None, DataTypeMetaData.KEY_STATE,
                                                               DataTypeMetaData.KEY_SUBJECT, None)
        return datatypes


class GetOperationsInProjectResource(Resource):

    def __init__(self):
        self.project_service = ProjectService()

    def get(self, project_id):
        _, _, operations, _ = self.project_service.retrieve_project_full(project_id)
        final_dict = dict()
        for operation in operations:
            dict_user = dict(
                id=operation['id'],
                user_id=operation['user'].id,
                algorithm_id=operation['algorithm'].id,
                group=operation['group'],
                gid=operation['gid'],
                create_date=operation['create'],
                start_date=operation['start'],
                completion_date=operation['complete'],
                status=operation['status'],
                visible=operation['visible']
            )

            final_dict[operation['gid']] = dict_user
        return jsonify(({'operations': final_dict}))


class GetOperationsForDatatypeResource(Resource):

    def __init__(self):
        self.flow_service = FlowService()

    def get(self, guid):
        categories = dao.get_launchable_categories()
        filtered_adapters = self.flow_service.get_filtered_adapters(guid, categories)
        operations = dict()

        for i in range(len(filtered_adapters)):
            operations[str(i)] = json.dumps(filtered_adapters[i].__dict__, cls=CustomEncoder)

        return operations

