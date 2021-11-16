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

from tvb.basic.neotraits.api import List, Int, EnumAttr, TVBEnum
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.neotraits.forms import TraitUploadField, SimpleLabelField, MultiSelectField, SelectField, IntField
from tvb.core.neotraits.view_model import ViewModel, Str
from tvb.storage.storage_interface import StorageInterface
from tvb.core.neocom import h5


class OutputVerbosityLevelsEnum(TVBEnum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4


class ParcellationOptionsEnum(TVBEnum):
    AAL_PARC = "aal"
    AAL2_PARC = "aal2"
    BRAINNETOME_PARC = "brainnetome246fs"
    CRADDOCK200_PARC = "craddock200"
    CRADDOCK400_PARC = "craddock400"
    DESIKAN_PARC = "desikan"
    DESTRIEUX_PARC = "destrieux"
    HCPMMP1_PARC = "hcpmmp1"
    PERRY512_PARC = "perry512"
    YEO7fs_PARC = "yeo7fs"
    YEO7mni_PARC = "yeo7mni"
    YEO17fs_PARC = "yeo17fs"
    YEO17mni_PARC = "yeo17mni"


class AnalysisLevelsEnum(TVBEnum):
    PREPROC_LEVEL = "preproc"
    PARTICIPANT_LEVEL = "participant"
    GROUP_LEVEL = "group"


class IPPipelineCreatorModel(ViewModel):
    mri_data = Str(
        label='Select MRI data for upload'
    )

    output_verbosity = EnumAttr(
        label="Select Output Verbosity",
        default=OutputVerbosityLevelsEnum.LEVEL_1,
        doc="""Select the verbosity of script output."""
    )

    analysis_level = EnumAttr(
        label="Select Analysis Level",
        default=AnalysisLevelsEnum.PREPROC_LEVEL,
        doc="""Select the analysis level that the pipeline will be launched on."""
    )

    parcellation = EnumAttr(
        label="Select Parcellation",
        default=ParcellationOptionsEnum.AAL_PARC,
        doc="""The choice of connectome parcellation scheme (compulsory for participant-level analysis)"""
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
        self.output_verbosity = SelectField(IPPipelineCreatorModel.output_verbosity, name='output_verbosity')
        self.pipeline_steps_label = SimpleLabelField("Configure pipeline steps")

        self.step1_choice = MultiSelectField(List(of=str, label="Step 1", choices=('Run step 1: fmriprep',),
                                                  default=(), required=False))
        self.parameters = MultiSelectField(IPPipelineCreatorModel.step_1_parameters)

        self.step2_choice = MultiSelectField(List(of=str, label="Step 2", choices=('Run step 2: mrtrix3',),
                                                  default=(), required=False))
        self.analysis_level = SelectField(IPPipelineCreatorModel.analysis_level, name='analysis_level')
        self.parcellation = SelectField(IPPipelineCreatorModel.parcellation, name='parcellation')
        self.stream_lines = IntField(IPPipelineCreatorModel.stream_lines)

        self.step3_choice = MultiSelectField(List(of=str, label="Step 3",
                                                  choices=('Run step 3: freesurfer',), default=(), required=False))
        self.step4_choice = MultiSelectField(List(of=str, label="Step 4",
                                                  choices=('Run step 4: tvb-pipeline-converter',),
                                                  default=(), required=False))

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
