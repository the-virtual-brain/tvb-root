# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
from tvb.core.entities.model.model_datatype import DataType
from tvb.core.entities.model.model_operation import Algorithm
from tvb.core.entities.model.model_project import User, Project
from tvb.interfaces.rest.commons.decoders import CustomDecoder


class BaseDto:
    def update(self, kwargs):
        # type: ({}) -> None
        """
        This method is setting object fields starting from a dictionary
        :param kwargs: fields dictionary (rest server response)
        """
        for key, value in kwargs.items():
            kwargs[key] = CustomDecoder.custom_hook(value)
        self.__dict__.update(kwargs)


class UserDto(BaseDto):
    def __init__(self, user=None, **kwargs):
        # type: (User, {}) -> None
        """
        Create an UserDto instance starting from an User entity or from a dictionary.
        For an UserDto object the unique identifier is the username field.
        :param user: DB User model
        :param kwargs: dictionary (rest server response)
        """
        self.update(kwargs)
        if user is not None:
            self.gid = user.gid
            self.display_name = user.display_name
            self.email = user.email


class ProjectDto(BaseDto):
    def __init__(self, project=None, **kwargs):
        # type: (Project, {}) -> None
        """
        Create a ProjectDto instance starting from a Project entity or from a dictionary.
        For an ProjectDto object the unique identifier is the gid field.
        :param project: DB Project model
        :param kwargs: dictionary (rest server response)
        """
        self.update(kwargs)
        if project is not None:
            self.gid = project.gid
            self.name = project.name
            self.description = project.description
            self.last_updated = project.last_updated
            self.version = project.version


class OperationDto(BaseDto):
    def __init__(self, operation=None, **kwargs):
        # type: ({}, {}) -> None
        """
        Create an OperationDto instance starting from an operations dictionary or from another dictionary.
        For an OperationDto object the unique identifier is the gid field.
        :param operation: dictionary computed from db
        :param kwargs: dictionary (rest server response)
        """
        self.update(kwargs)
        if hasattr(self, 'algorithm_dto'):
            self.algorithm_dto = AlgorithmDto(None, **self.algorithm_dto)
        if operation is not None:
            self.user_gid = operation['user'].gid
            self.algorithm_dto = AlgorithmDto(operation['algorithm'])
            self.group = operation['group']
            self.gid = operation['gid']
            self.create_date = operation['create']
            self.start_date = operation['start']
            self.completion_date = operation['complete']
            self.status = operation['status']
            self.visible = operation['visible']

    @property
    def displayname(self):
        return self.algorithm_dto.displayname

    @property
    def description(self):
        return self.algorithm_dto.description

class AlgorithmDto(BaseDto):
    def __init__(self, algorithm=None, **kwargs):
        # type: (Algorithm, {}) -> None
        """
        Create an AlgorithmDto instance starting from an Algorithm entity or from a dictionary.
        For an OperationDto object the unique identifier is tuple (module,classname).
        :param algorithm: DB Algorithm model
        :param kwargs: dictionary (rest server response)
        """
        self.update(kwargs)
        if algorithm is not None:
            self.module = algorithm.module
            self.classname = algorithm.classname
            self.displayname = algorithm.displayname
            self.description = algorithm.description


class DataTypeDto(BaseDto):
    def __init__(self, datatype=None, **kwargs):
        # type: (DataType, {}) -> None
        """
        Create a DataTypeDto instance starting from a DataType entity or from a dictionary.
        For an DataTypeDto object the unique identifier is the gid field.
        :param datatype: DB DataType model
        :param kwargs: dictionary (rest server response)
        """
        self.update(kwargs)
        if datatype is not None:
            self.gid = datatype.gid
            self.name = datatype.display_name
            self.type = datatype.display_type
            self.create_date = datatype.create_date
            self.subject = datatype.subject
