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

"""
Entities for Generic DataTypes, Links and Groups of DataTypes are defined here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Yann Gordon <yann@tvb.invalid>
"""
import json
import numpy
import typing
from datetime import datetime
from copy import copy
from sqlalchemy.orm import relationship, backref
from sqlalchemy import Boolean, Integer, String, Float, Column, ForeignKey
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits
from tvb.core.entities.generic_attributes import GenericAttributes
from tvb.core.entities.model.model_project import Project
from tvb.core.entities.model.model_operation import Operation, OperationGroup
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.neotraits.db import HasTraitsIndex, Base, from_ndarray


LOG = get_logger(__name__)

FILTER_CATEGORIES = {'DataType.subject': {'display': 'Subject', 'type': 'string',
                                          'operations': ['!=', '==', 'like', 'in', 'not in']},
                     'DataType.state': {'display': 'State', 'type': 'string',
                                        'operations': ['!=', '==', 'in', 'not in']},
                     'DataType.disk_size': {'display': 'Disk Size (KB)', 'type': 'int',
                                            'operations': ['<', '==', '>']},
                     'DataType.user_tag_1': {'display': 'Tag 1', 'type': 'string',
                                             'operations': ['!=', '==', 'like']},
                     'DataType.user_tag_2': {'display': 'Tag 2', 'type': 'string',
                                             'operations': ['!=', '==', 'like']},
                     'DataType.user_tag_3': {'display': 'Tag 3', 'type': 'string',
                                             'operations': ['!=', '==', 'like']},
                     'DataType.user_tag_4': {'display': 'Tag 4', 'type': 'string',
                                             'operations': ['!=', '==', 'like']},
                     'DataType.user_tag_5': {'display': 'Tag 5', 'type': 'string',
                                             'operations': ['!=', '==', 'like']},
                     'BurstConfiguration.name': {'display': 'Simulation name', 'type': 'string',
                                                 'operations': ['==', '!=', 'like']},
                     'Operation.user_group': {'display': 'Operation Tag', 'type': 'string',
                                              'operations': ['==', '!=', 'like']},
                     'Operation.start_date': {'display': 'Start date', 'type': 'date',
                                              'operations': ['!=', '<', '>']},
                     'Operation.completion_date': {'display': 'Completion date', 'type': 'date',
                                                   'operations': ['!=', '<', '>']}}


class DataType(HasTraitsIndex):
    """ 
    Base class for DB storage of Types.
    DataTypes, are the common language between Visualizers, 
    Simulator and Analyzers.

    """

    id = Column(Integer, ForeignKey(HasTraitsIndex.id), primary_key=True)
    type = Column(String)  # Name of class inheriting from current type
    module = Column(String)
    subject = Column(String)
    state = Column(String)  # RAW, INTERMEDIATE, FINAL
    visible = Column(Boolean, default=True)
    invalid = Column(Boolean, default=False)
    is_nan = Column(Boolean, default=False)
    disk_size = Column(Integer, default=0)
    user_tag_1 = Column(String)  # Name used by framework and perpetuated from a DataType to derived entities.
    user_tag_2 = Column(String)
    user_tag_3 = Column(String)
    user_tag_4 = Column(String)
    user_tag_5 = Column(String)

    # GID of a burst in which current dataType was generated
    # Native burst-results are referenced from a workflowSet as well
    # But we also have results generated afterwards from TreeBurst tab.
    fk_parent_burst = Column(String(32), ForeignKey(BurstConfiguration.gid, ondelete="SET NULL"), nullable=True)
    _parent_burst = relationship(BurstConfiguration, foreign_keys=fk_parent_burst,
                                 primaryjoin=BurstConfiguration.gid == fk_parent_burst, cascade='none')

    # it should be a reference to a DataTypeGroup, but we can not create that FK
    # because this two tables (DATA_TYPES, DATA_TYPES_GROUPS) will reference each
    # other mutually and SQL-Alchemy complains about that.
    fk_datatype_group = Column(Integer, ForeignKey('DataType.id'))

    fk_from_operation = Column(Integer, ForeignKey('OPERATIONS.id', ondelete="CASCADE"))
    parent_operation = relationship(Operation, backref=backref("DATA_TYPES", order_by=id, cascade="all,delete"))

    # Transient info
    fixed_generic_attributes = False

    def get_extra_info(self):
        return {}

    def __init__(self, gid=None, **kwargs):

        self.gid = gid
        self.type = self.__class__.__name__
        self.module = self.__class__.__module__

        try:
            self.__initdb__(**kwargs)
        except Exception as exc:
            LOG.warning('Could not perform __initdb__: %r', exc)
        super(DataType, self).__init__()

    def __initdb__(self, subject='', state=None, operation_id=None, fk_parent_burst=None, disk_size=None,
                   user_tag_1=None, user_tag_2=None, user_tag_3=None, user_tag_4=None, user_tag_5=None, **_):
        """Set attributes"""
        self.subject = subject
        self.state = state
        self.fk_from_operation = operation_id
        self.user_tag_1 = user_tag_1
        self.user_tag_2 = user_tag_2
        self.user_tag_3 = user_tag_3
        self.user_tag_4 = user_tag_4
        self.user_tag_5 = user_tag_5
        self.disk_size = disk_size
        self.fk_parent_burst = fk_parent_burst

    def fill_from_h5(self, h5_file):
        LOG.warning("fill_from_h5 for: {}".format(type(self)))
        self.gid = h5_file.gid.load().hex


    def after_store(self):
        """
        Put here code (as a trigger after storage) to be executed by
        ABCAdapter after the current DT is stored in DB
        """
        pass

    def __repr__(self):
        msg = "<DataType(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)>"
        return msg % (str(self.id), self.gid, self.type, self.module,
                      self.subject, self.state, str(self.fk_parent_burst),
                      self.user_tag_1, self.user_tag_2, self.user_tag_3, self.user_tag_4, self.user_tag_5)

    @property
    def display_type(self):
        return self.type.replace("Index", "")

    @property
    def display_name(self):
        """
        To be implemented in each sub-class which is about to be displayed in UI, 
        and return the text to appear.
        """
        display_name = self.display_type
        for tag in [self.user_tag_1, self.user_tag_2, self.user_tag_3, self.user_tag_4, self.user_tag_5]:
            if tag is not None and len(tag) > 0:
                display_name += " - " + str(tag)
        return display_name

    @property
    def summary_info(self):
        # type: () -> typing.Dict[str, str]

        ret = {}
        if self.title:
            ret['Title'] = str(self.title)

        columns = self._get_table_columns()
        for attr_name in columns:
            try:
                if "id" == attr_name:
                    continue
                name = attr_name.title().replace("Fk_", "Linked ").replace("_", " ")
                attr_value = getattr(self, attr_name)
                ret[name] = str(attr_value)
            except Exception:
                pass
        return ret

    @property
    def is_ts(self):
        return hasattr(self, 'time_series_type')

    def _get_table_columns(self):
        columns = self.__table__.columns.keys()
        if type(self).__bases__[0] is DataType:
            return columns
        # Consider the immediate superclass only, as for now we have
        # - most of *Index classes directly inheriting from DataType
        # - except the ones with one intermediate: DataTypeMatrix or TimeSeriesIndex
        base_table_columns = type(self).__bases__[0].__table__.columns.keys()
        columns.extend(base_table_columns)
        return columns

    @staticmethod
    def accepted_filters():
        """
        Return accepted UI filters for current DataType.
        """
        return copy(FILTER_CATEGORIES)

    def fill_from_has_traits(self, has_traits):
        # type: (HasTraits) -> None
        self.gid = has_traits.gid.hex

    def fill_from_generic_attributes(self, attrs):
        # type: (GenericAttributes) -> None
        self.subject = attrs.subject
        self.state = attrs.state
        self.user_tag_1 = attrs.user_tag_1
        self.user_tag_2 = attrs.user_tag_2
        self.user_tag_3 = attrs.user_tag_3
        self.user_tag_4 = attrs.user_tag_4
        self.user_tag_5 = attrs.user_tag_5
        self.fk_parent_burst = attrs.parent_burst
        self.is_nan = attrs.is_nan
        self.visible = attrs.visible
        self.create_date = attrs.create_date or datetime.now()


class DataTypeMatrix(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)
    subtype = Column(String, nullable=True)

    ndim = Column(Integer, default=0)
    shape = Column(String, nullable=True)
    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)
    array_is_finite = Column(Boolean, default=True)
    array_has_complex = Column(Boolean, default=False)
    # has_volume_mapping will currently be changed for ConnectivityMeasureIndex subclass
    has_volume_mapping = Column(Boolean, nullable=False, default=False)
    # currently has_valid_time_series is False only when importing Correlation Coefficients via BIDS
    has_valid_time_series = Column(Boolean, nullable=False, default=True)

    def fill_from_has_traits(self, datatype):
        super(DataTypeMatrix, self).fill_from_has_traits(datatype)
        self.subtype = datatype.__class__.__name__

        if hasattr(datatype, "array_data"):
            self.array_has_complex = numpy.iscomplex(datatype.array_data).any().item()

            if not self.array_has_complex:
                self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.array_data)

            self.array_is_finite = numpy.isfinite(datatype.array_data).all().item()
            self.shape = json.dumps(datatype.array_data.shape)
            self.ndim = len(datatype.array_data.shape)

    @property
    def parsed_shape(self):
        try:
            return tuple(json.loads(self.shape))
        except:
            return ()


class DataTypeGroup(DataType):
    """
    All the DataTypes resulted from an operation group will be part from a DataType group.
    """
    __tablename__ = 'DATA_TYPES_GROUPS'

    id = Column('id', Integer, ForeignKey('DataType.id', ondelete="CASCADE"), primary_key=True)
    count_results = Column(Integer)
    no_of_ranges = Column(Integer, default=0)  # Number of ranged parameters
    fk_operation_group = Column(Integer, ForeignKey('OPERATION_GROUPS.id', ondelete="CASCADE"))

    parent_operation_group = relationship(OperationGroup, backref=backref("DATA_TYPES_GROUPS", cascade="delete"))

    def __init__(self, operation_group, **kwargs):
        super(DataTypeGroup, self).__init__(**kwargs)

        self.fk_operation_group = operation_group.id
        self.count_results = 0

        if operation_group.range3 is not None:
            self.no_of_ranges = 3
        elif operation_group.range2 is not None:
            self.no_of_ranges = 2
        elif operation_group.range1 is not None:
            self.no_of_ranges = 1
        else:
            self.no_of_ranges = 0

    @staticmethod
    def is_data_a_group(data):
        """
        Checks if the provided data, ready for export is a DataTypeGroup or not
        """
        return isinstance(data, DataTypeGroup)


class Links(Base):
    """
    Class used to handle shortcuts from one DataType to another project.
    """
    __tablename__ = 'LINKS'

    id = Column(Integer, primary_key=True)
    fk_to_project = Column(Integer, ForeignKey('PROJECTS.id', ondelete="CASCADE"))
    fk_from_datatype = Column(Integer, ForeignKey('DataType.id', ondelete="CASCADE"))

    referenced_project = relationship(Project, backref=backref('LINKS', order_by=id, cascade="delete, all"))
    referenced_datatype = relationship(DataType, backref=backref('LINKS', order_by=id, cascade="delete, all"))


    def __init__(self, from_datatype, to_project):
        self.fk_from_datatype = from_datatype
        self.fk_to_project = to_project


    def __repr__(self):
        return '<Link(%d, %d)>' % (self.fk_from_datatype, self.fk_to_project)



class MeasurePointsSelection(Base):
    """
    Interest area.
    A subset of nodes from a Connectivity or Sensors.
    """

    __tablename__ = "MEASURE_POINTS_SELECTIONS"

    id = Column(Integer, primary_key=True)
    # Unique name /DataType/Project, to be displayed in selector UI:
    ui_name = Column(String)
    # JSON with node indices in current selection (0-based):
    selected_nodes = Column(String)
    # A Connectivity of Sensor GID, Referring to the entity that this selection was produced for:
    fk_datatype_gid = Column(String(32), ForeignKey('HasTraitsIndex.gid', ondelete="CASCADE"))
    # Current Project the selection was defined in:
    fk_in_project = Column(Integer, ForeignKey('PROJECTS.id', ondelete="CASCADE"))


    def __init__(self, ui_name, selected_nodes, datatype_gid, project_id):
        self.ui_name = ui_name
        self.selected_nodes = selected_nodes
        self.fk_in_project = project_id
        self.fk_datatype_gid = datatype_gid


    def __repr__(self):
        return '<Selection(%s, %s, for %s)>' % (self.ui_name, self.selected_nodes, self.fk_datatype_gid)


class StoredPSEFilter(Base):
    """
    Interest Area
    PSE Viewer Filter tool, specific filter configuration user inputs to be stored from multiple elements
    """
    __tablename__ = "PSE_FILTERS"

    id = Column(Integer, primary_key=True)
    # Unique name /DataType, to be displayed in selector UI:
    ui_name = Column(String)
    # A DataType Group GID, Referring to the Group that this filter was stored for:
    fk_datatype_gid = Column(String(32), ForeignKey('HasTraitsIndex.gid', ondelete="CASCADE"))

    threshold_value = Column(Float)

    applied_on = Column(String)

    def __init__(self, ui_name, datatype_gid, threshold_value, applied_on):
        self.ui_name = ui_name
        self.fk_datatype_gid = datatype_gid
        self.threshold_value = threshold_value
        self.applied_on = applied_on


    def __repr__(self):
        return '<StoredPSEFilter(%s, for %s, %s, %s)>' % (self.ui_name, self.fk_datatype_gid,
                                                          self.threshold_value, self.applied_on)
