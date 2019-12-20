class UserDto:
    def __init__(self, user):
        self.username = user.username
        self.email = user.email
        self.validated = user.validated
        self.role = user.role


class ProjectDto:
    def __init__(self, project):
        self.gid = project.gid
        self.name = project.name
        self.description = project.description
        self.gid = project.gid
        self.version = project.version


class OperationDto:
    def __init__(self, operation):
        self.user_id = operation['user'].id
        self.algorithm_id = operation['algorithm'].id
        self.group = operation['group']
        self.gid = operation['gid']
        self.create_date = operation['create']
        self.start_date = operation['start']
        self.completion_date = operation['complete']
        self.status = operation['status']
        self.visible = operation['visible']


class AlgorithmDto:
    def __init__(self, algorithm):
        self.module = algorithm.module
        self.classname = algorithm.classname
        self.displayname = algorithm.displayname
        self.description = algorithm.description


class DataTypeDto:
    def __init__(self, datatype):
        self.gid = datatype.gid
        self.name = datatype.display_name
        self.type = datatype.display_type
