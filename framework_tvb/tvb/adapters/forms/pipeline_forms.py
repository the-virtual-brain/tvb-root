# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from tvb.core.neotraits.forms import Form, SelectField, IntField, StrField
from tvb.core.neotraits.view_model import Str
from tvb.basic.neotraits.api import EnumAttr, TVBEnum, Int, TupleEnum, HasTraits, Attr
from tvb.core.pipeline.analysis_levels import PreprocAnalysisLevel, ParticipantAnalysisLevel, GroupAnalysisLevel, \
    ParcellationOptionsEnum, TemplateRegOptionsEnum


class PipelineForm(Form):
    @staticmethod
    def get_subform_key():
        return "PIPELINE"


class CommonPipelineForm(PipelineForm):
    """
    Contains options that are relevant to both 'preproc' level and 'participant' level
    """

    def __init__(self):
        super(CommonPipelineForm, self).__init__()
        self.t1w_preproc_path = StrField(Attr(field_type=str, label='t1w preproc', required=False,
                                              doc="""Provide a path by which pre-processed T1-weighted image data may
                                              be found for the processed participant(s) / session(s) """),
                                         name="t1w_preproc")

    def fill_trait(self, datatype):
        datatype.parameters['t1w_preproc_path'] = self.t1w_preproc_path.value

    def fill_from_trait(self, trait):
        self.t1w_preproc_path.data = trait.parameters['t1w_preproc_path']


class ParticipantPipelineForm(CommonPipelineForm):
    """
    Contains options that are relevant only for the 'participant' analysis level
    """

    def __init__(self):
        super(ParticipantPipelineForm, self).__init__()

        self.parcellation = SelectField(EnumAttr(label="Select Parcellation", default=ParcellationOptionsEnum.AAL_PARC,
                                                 doc="""The choice of connectome parcellation scheme (compulsory for 
                                                 participant-level analysis)"""), name='parcellation')

        self.stream_lines = IntField(Int(label="Number of stream lines", required=False, default=1,
                                         doc="""The number of streamlines to generate for each subject (will be 
                                         determined heuristically if not explicitly set)."""), name="stream_lines")

        self.template_reg = SelectField(EnumAttr(label="Template Reg", default=TemplateRegOptionsEnum.ANTS_TEMPLATE_REG,
                                                 doc="""The choice of registration software for mapping subject to
                                                  template space."""), name="template_reg")

    def fill_trait(self, datatype):
        datatype.parameters['t1w_preproc_path'] = self.t1w_preproc_path.value
        datatype.parameters['parcellation'] = self.parcellation.value
        datatype.parameters['stream_lines'] = self.stream_lines.value
        datatype.parameters['template_reg'] = self.template_reg.value

    def fill_from_trait(self, trait):
        self.t1w_preproc_path.data = trait.parameters['t1w_preproc_path']
        self.parcellation.data = trait.parameters['parcellation']
        self.stream_lines.data = trait.parameters['stream_lines']
        self.template_reg.data = trait.parameters['template_reg']


class GroupPipelineForm(PipelineForm):
    """
    Contains options that are relevant only for the 'group' analysis level.
    """

    def __init__(self):
        super(GroupPipelineForm, self).__init__()

        self.group_participant_label = StrField(Str('Participant label(s)', label="Participant label",
                                                    doc="Select one or multiple participant labels by delimiting"
                                                        " the names with commas."), name='group_participant_label')
        self.session_label = StrField(Str('Session label', label="Session label",
                                          doc="The session(s) within each participant that should be analyzed."),
                                      name='session_label')

    def fill_trait(self, datatype):
        datatype.parameters['group_participant_label'] = self.group_participant_label.value
        datatype.parameters['session_label'] = self.session_label.value

    def fill_from_trait(self, trait):
        self.group_participant_label.data = trait.parameters['group_participant_label']
        self.session_label.data = trait.parameters['session_label']


class IPPipelineAnalysisLevelsEnum(TupleEnum):
    PREPROC_LEVEL = (PreprocAnalysisLevel, "preproc")
    PARTICIPANT_LEVEL = (ParticipantAnalysisLevel, "participant")
    GROUP_LEVEL = (GroupAnalysisLevel, "group")


def get_form_for_analysis_level(analysis_level):
    al_form = {
        PreprocAnalysisLevel: CommonPipelineForm,
        ParticipantAnalysisLevel: ParticipantPipelineForm,
        GroupAnalysisLevel: GroupPipelineForm
    }

    return al_form[analysis_level]
