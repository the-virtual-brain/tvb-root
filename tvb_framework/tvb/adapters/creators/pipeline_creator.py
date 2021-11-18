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

import os

from tvb.adapters.forms.form_methods import PIPELINE_KEY
from tvb.adapters.forms.pipeline_forms import IPPipelineAnalysisLevelsEnum, CommonPipelineForm, PreprocAnalysisLevel, \
    PipelineAnalysisLevel
from tvb.basic.neotraits.api import List, Int, EnumAttr, TVBEnum, Attr
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.neotraits.forms import TraitUploadField, SimpleLabelField, MultiSelectField, SelectField, StrField, \
    BoolField
from tvb.core.neotraits.view_model import ViewModel, Str
from tvb.storage.storage_interface import StorageInterface
from tvb.core.neocom import h5


class OutputVerbosityLevelsEnum(TVBEnum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4


class IPPipelineCreatorModel(ViewModel):
    mri_data = Str(
        label='Select MRI data for upload'
    )

    participant_label = Str(
        label='Participant Label',
        doc=r"""The filename part after "sub-" in BIDS format"""
    )

    step1_choice = Attr(
        field_type=bool,
        label="Run step 1: MRtrix3",
    )

    step2_choice = Attr(
        field_type=bool,
        label="Run step 2: fmriprep",
    )

    step3_choice = Attr(
        field_type=bool,
        label="Run step 3: freesurfer",
    )

    step4_choice = Attr(
        field_type=bool,
        label="Run step 4: tvb-pipeline-converter",
    )

    output_verbosity = EnumAttr(
        label="Select Output Verbosity",
        default=OutputVerbosityLevelsEnum.LEVEL_1,
        doc="""Select the verbosity of script output."""
    )

    analysis_level = Attr(
        field_type=PipelineAnalysisLevel,
        label="Analysis Level",
        required=True,
        doc="""Select the analysis level that the pipeline will be launched on.""",
        default=PreprocAnalysisLevel()
    )

    stream_lines = Int(
        label="Number of stream lines",
        required=False,
        default=1,
        doc="""The number of streamlines to generate for each subject (will be determined heuristically
         if not explicitly set)."""
    )

    step_1_parameters = List(
        of=str,
        label='Parameters',
        choices=('6 DoF', 'MNI normalization'),
        required=False
    )


KEY_PIPELINE = "ip-pipeline"


class IPPipelineCreatorForm(ABCAdapterForm):

    def __init__(self):
        super(IPPipelineCreatorForm, self).__init__()

        self.pipeline_job = SimpleLabelField("Pipeline Job1")
        self.mri_data = TraitUploadField(IPPipelineCreatorModel.mri_data, '.zip', 'mri_data')
        self.participant_label = StrField(IPPipelineCreatorModel.participant_label)
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
        self.parameters = MultiSelectField(IPPipelineCreatorModel.step_1_parameters)

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
