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
import json
import os

from tvb.adapters.forms.form_methods import PIPELINE_KEY
from tvb.adapters.forms.pipeline_forms import IPPipelineAnalysisLevelsEnum, CommonPipelineForm, \
    ParticipantPipelineForm, GroupPipelineForm, get_form_for_analysis_level
from tvb.basic.neotraits.api import List, Int, EnumAttr, TVBEnum, Attr
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.neotraits.forms import TraitUploadField, SimpleLabelField, MultiSelectField, SelectField, StrField, \
    BoolField, FormField, IntField
from tvb.core.neotraits.view_model import ViewModel, Str
from tvb.core.pipeline.analysis_levels import PipelineAnalysisLevel, PreprocAnalysisLevel, ParticipantAnalysisLevel, \
    GroupAnalysisLevel
from tvb.storage.storage_interface import StorageInterface
from tvb.core.neocom import h5


class OutputVerbosityLevelsEnum(str, TVBEnum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4


class IPPipelineCreatorModel(ViewModel):
    PIPELINE_CONFIG_FILE = "pipeline_configurations.json"

    mri_data = Str(
        label='Select MRI data for upload',
        default='enter path here'
    )

    participant_label = Str(
        label='Participant Label',
        default='sub-Con03',
        doc=r"""The filename part after "sub-" in BIDS format"""
    )

    nr_of_cpus = Int(
        label="Number of cpus",
        required=False,
        default=1,
        doc="""The number of cpus that will be used for launching the pipeline."""
    )

    estimated_time = Int(
        label="Estimated runtime (hours)",
        required=False,
        default=1,
        doc="""Estimated duration of running the pipeline expressed in hours."""
    )

    step1_choice = Attr(
        field_type=bool,
        default=False,
        label="Run step 1: MRtrix3",
    )

    output_verbosity = EnumAttr(
        label="Select Output Verbosity",
        default=OutputVerbosityLevelsEnum.LEVEL_1,
        doc="""Select the verbosity of script output."""
    )

    analysis_level = Attr(
        field_type=PipelineAnalysisLevel,
        label="Analysis Level",
        doc="""Select the analysis level that the pipeline will be launched on.""",
        default=PreprocAnalysisLevel()
    )

    step2_choice = Attr(
        field_type=bool,
        default=False,
        label="Run step 2: fmriprep",
    )

    skip_bids = Attr(
        field_type=bool,
        default=False,
        doc="Assume the input dataset is BIDS compliant and skip the validation.",
        label="Skip Bids Validation"
    )

    anat_only = Attr(
        field_type=bool,
        default=False,
        doc="Run anatomical workflows only.",
        label="Anat Only"
    )

    no_reconall = Attr(
        field_type=bool,
        default=False,
        doc="Disable FreeSurfer surface preprocessing.",
        label="No Reconall"
    )

    step2_parameters = List(
        of=str,
        label='Parameters',
        choices=('6 DoF', 'MNI normalization'),
        required=False
    )

    step3_choice = Attr(
        field_type=bool,
        default=False,
        label="Run step 3: freesurfer",
    )

    step4_choice = Attr(
        default=False,
        field_type=bool,
        label="Run step 4: tvb-pipeline-converter",
    )

    def to_json(self, storage_path):
        pipeline_config = {
            'mri_data': os.path.basename(self.mri_data),
            'participant_label': self.participant_label,
            'nr_of_cpus': self.nr_of_cpus,
            'estimated_time': self.estimated_time,
            'step1_choice': self.step1_choice,
            'step1_parameters': {
                                'output_verbosity': self.output_verbosity,
                                'analysis_level': str(self.analysis_level),
                                'analysis_level_config': self.analysis_level.parameters,
                                },
            'step2_choice': self.step2_choice,
            'step2_parameters': {
                                'skip_bids': self.skip_bids,
                                'anat_only': self.anat_only,
                                'no_reconall': self.no_reconall,
                                },
            'step3_choice': self.step3_choice,
            'step4_choice': self.step4_choice,
        }

        with open(os.path.join(storage_path, self.PIPELINE_CONFIG_FILE), 'w') as f:
            json.dump(pipeline_config, f)


KEY_PIPELINE = "ip-pipeline"


class IPPipelineCreatorForm(ABCAdapterForm):

    def __init__(self):
        super(IPPipelineCreatorForm, self).__init__()

        self.pipeline_job = SimpleLabelField("Pipeline Job1")
        self.mri_data = TraitUploadField(IPPipelineCreatorModel.mri_data, '.zip', 'mri_data')
        self.participant_label = StrField(IPPipelineCreatorModel.participant_label)
        self.nr_of_cpus = IntField(IPPipelineCreatorModel.nr_of_cpus, name='number_of_cpus')
        self.estimated_time = IntField(IPPipelineCreatorModel.estimated_time, name='estimated_time')
        self.pipeline_steps_label = SimpleLabelField("Configure pipeline steps")

        self.step1_choice = BoolField(IPPipelineCreatorModel.step1_choice)
        self.output_verbosity = SelectField(IPPipelineCreatorModel.output_verbosity, name='output_verbosity')
        self.analysis_level = SelectField(EnumAttr(field_type=IPPipelineAnalysisLevelsEnum,
                                                   label="Select Analysis Level", required=True,
                                                   default=IPPipelineAnalysisLevelsEnum.PREPROC_LEVEL.instance,
                                                   doc="""Select the analysis level that the pipeline will be launched
                                                    on."""), name='analysis_level', subform=CommonPipelineForm,
                                          session_key=KEY_PIPELINE, form_key=PIPELINE_KEY)

        self.step2_choice = BoolField(IPPipelineCreatorModel.step2_choice)
        self.skip_bids = BoolField(IPPipelineCreatorModel.skip_bids)
        self.anat_only = BoolField(IPPipelineCreatorModel.anat_only)
        self.no_reconall = BoolField(IPPipelineCreatorModel.no_reconall)
        self.step2_parameters = MultiSelectField(IPPipelineCreatorModel.step2_parameters)

        self.step3_choice = BoolField(IPPipelineCreatorModel.step3_choice)
        self.step4_choice = BoolField(IPPipelineCreatorModel.step4_choice)

    @staticmethod
    def get_required_datatype():
        pass

    @staticmethod
    def get_filters():
        pass

    @staticmethod
    def get_input_name():
        return None

    @staticmethod
    def get_view_model():
        return IPPipelineCreatorModel

    def fill_from_trait(self, trait):
        # type: (IPPipelineCreatorModel) -> None
        super(IPPipelineCreatorForm, self).fill_from_trait(trait)
        analysis_level = trait.analysis_level
        self.analysis_level.data = type(analysis_level)
        self.analysis_level.subform_field = FormField(get_form_for_analysis_level(type(analysis_level)),
                                                      'subform_analysis_level')
        self.analysis_level.subform_field.form.fill_from_trait(analysis_level)

    def fill_trait(self, datatype):
        super(IPPipelineCreatorForm, self).fill_trait(datatype)
        if self.analysis_level.data == IPPipelineAnalysisLevelsEnum.PREPROC_LEVEL:
            datatype.analysis_level = PreprocAnalysisLevel()
        elif self.analysis_level.data == IPPipelineAnalysisLevelsEnum.PARTICIPANT_LEVEL:
            datatype.analysis_level = ParticipantAnalysisLevel()
        else:
            datatype.analysis_level = GroupAnalysisLevel()

        self.analysis_level.subform_field.form.fill_trait(datatype.analysis_level)

    def fill_from_post(self, form_data):
        super(IPPipelineCreatorForm, self).fill_from_post(form_data)
        if form_data['analysis_level'] == 'preproc':
            self.analysis_level.subform_field.form = CommonPipelineForm()
        elif form_data['analysis_level'] == 'participant':
            self.analysis_level.subform_field.form = ParticipantPipelineForm()
        else:
            self.analysis_level.subform_field.form = GroupPipelineForm()

        self.analysis_level.subform_field.form.fill_from_post(form_data)


class IPPipelineCreator(ABCAdapter):
    _ui_name = "Launch Image Preprocessing Pipeline"
    _ui_description = "Launch Image Preprocessing Pipeline from tvb-web when it is deployed to EBRAINS"
    PIPELINE_DATASET_FILE = "pipeline_dataset.zip"

    def get_form_class(self):
        return IPPipelineCreatorForm

    def get_output(self):
        return []

    def get_required_disk_size(self, view_model):
        return -1

    def get_required_memory_size(self, view_model):
        return -1

    def launch(self, view_model):
        # type: (IPPipelineCreatorModel) -> []
        storage_path = self.get_storage_path()
        dest_path = os.path.join(storage_path, self.PIPELINE_DATASET_FILE)
        StorageInterface.copy_file(view_model.mri_data, dest_path)
        view_model.mri_data = dest_path
        h5.store_view_model(view_model, storage_path)

        view_model.to_json(storage_path)
