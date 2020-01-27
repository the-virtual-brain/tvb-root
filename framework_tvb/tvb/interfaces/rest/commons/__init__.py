from enum import Enum


class Strings(Enum):
    PAGE_NUMBER = "page"
    BASE_PATH = "api"


class RestNamespace(Enum):
    USERS = "/users"
    PROJECTS = "/projects"
    DATATYPES = "/datatypes"
    OPERATIONS = "/operations"
    SIMULATION = "/simulation"


class LinkPlaceholder(Enum):
    USERNAME = "username"
    PROJECT_GID = "project_gid"
    DATATYPE_GID = "datatype_gid"
    OPERATION_GID = "operation_gid"
    ALG_MODULE = "algorithm_module"
    ALG_CLASSNAME = "algorithm_classname"


class RestLink(Enum):
    # USERS
    PROJECTS_LIST = "/{" + LinkPlaceholder.USERNAME.value + "}/projects"

    # PROJECTS
    DATA_IN_PROJECT = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/data"
    OPERATIONS_IN_PROJECT = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/operations"

    # DATATYPES
    GET_DATATYPE = "/{" + LinkPlaceholder.DATATYPE_GID.value + "}"
    DATATYPE_OPERATIONS = "/{" + LinkPlaceholder.DATATYPE_GID.value + "}/operations"

    # OPERATIONS
    LAUNCH_OPERATION = "/{" + LinkPlaceholder.PROJECT_GID.value + "}/algorithm/{" + LinkPlaceholder.ALG_MODULE.value + "}/{" + LinkPlaceholder.ALG_CLASSNAME.value + "}"
    OPERATION_STATUS = "/{" + LinkPlaceholder.OPERATION_GID.value + "}/status"
    OPERATION_RESULTS = "/{" + LinkPlaceholder.OPERATION_GID.value + "}/results"

    # SIMULATION
    FIRE_SIMULATION = "/{" + LinkPlaceholder.PROJECT_GID.value + "}"

    def compute_url(self, include_namespace=False, values=None):
        if values is None:
            values = {}
        result = ""
        if include_namespace:
            for namespace, links in _namespace_url_dict.items():
                if self in links:
                    result += namespace.value
                    break
        result += self.value
        return result.format(**values)


_namespace_url_dict = {
    RestNamespace.USERS: [RestLink.PROJECTS_LIST],
    RestNamespace.PROJECTS: [RestLink.DATA_IN_PROJECT, RestLink.OPERATIONS_IN_PROJECT],
    RestNamespace.DATATYPES: [RestLink.GET_DATATYPE, RestLink.DATATYPE_OPERATIONS],
    RestNamespace.OPERATIONS: [RestLink.LAUNCH_OPERATION, RestLink.OPERATION_STATUS, RestLink.OPERATION_RESULTS],
    RestNamespace.SIMULATION: [RestLink.FIRE_SIMULATION]
}
