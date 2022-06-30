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
import shutil
import tempfile
import pathlib

import cherrypy._cpreqbody
from tvb.adapters.forms.pipeline_forms import IPPipelineAnalysisLevelsEnum, get_form_for_analysis_level, \
    PreprocPipelineForm
from tvb.basic.neotraits.api import List, Int, EnumAttr, TVBEnum, Attr
from tvb.basic.profile import TvbProfile
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.neotraits.forms import SimpleLabelField, SelectField, StrField, BoolField, FormField, \
    IntField, Form, ValidatedTraitUploadField
from tvb.core.neotraits.view_model import ViewModel, Str
from tvb.core.pipeline.analysis_levels import PipelineAnalysisLevel, PreprocAnalysisLevel, ParticipantAnalysisLevel, \
    GroupAnalysisLevel
from typing import List

from tvb.storage.storage_interface import StorageInterface
from tvb.core.neocom import h5

from tvb.basic.logger.builder import get_logger


_logger = get_logger(__name__)


MRI_DATA_FIELD_NAME = 'mri_data'
IGNORED_FILES = '.DS_Store'


class OutputVerbosityLevelsEnum(str, TVBEnum):
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4


class ParcellationOptionsEnum(str, TVBEnum):
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


class IPPipelineCreatorModel(ViewModel):
    PIPELINE_CONFIG_FILE = "pipeline_configurations.json"
    PIPELINE_DATASET_FILE = "pipeline_dataset.zip"

    mri_data = Str(
        label='Select MRI data for upload',
        default='enter path here'
    )

    participant_label = Str(
        label='Participant Label',
        default='sub-CON03',
        doc=r"""The filename part after "sub-" in BIDS format"""
    )

    session_label = Str(
        label='Session Label',
        default='ses-postop',
        doc="Subject subdirectory name"
    )

    task_label = Str(
        label='Task Label',
        default='rest'
    )

    parcellation = EnumAttr(
        label="Select Parcellation",
        default=ParcellationOptionsEnum.AAL_PARC,
        doc="""The choice of connectome parcellation scheme (compulsory for participant-level analysis)"""
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

    wall_time_step1 = Int(
        label="Estimated runtime - Step 1 (hours)",
        required=False,
        default=1,
        doc="""Estimated duration of step 1 of the pipeline expressed in hours."""
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

    wall_time_step2 = Int(
        label="Estimated runtime - Step 2 (hours)",
        required=False,
        default=1,
        doc="""Estimated duration of step 2 of the pipeline expressed in hours."""
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

    dof_6 = Attr(
        field_type=bool,
        required=False,
        label="6 DoF"
    )

    mni_normalization = Attr(
        field_type=bool,
        required=False,
        label="MNI normalization"
    )

    step3_choice = Attr(
        field_type=bool,
        default=False,
        label="Run step 3: tvb-pipeline-converter",
    )

    wall_time_step3 = Int(
        label="Estimated runtime - Step 3 (hours)",
        required=False,
        default=1,
        doc="""Estimated duration of step 3 of the pipeline expressed in hours."""
    )

    def to_json(self, storage_path):
        pipeline_config = {
            'mri_data': os.path.basename(self.mri_data),
            'participant_label': self.participant_label,
            'session_label': self.session_label,
            'task-label': self.task_label,
            'parcellation': self.parcellation,
            'nr_of_cpus': self.nr_of_cpus,
            'mrtrix': self.step1_choice,
            'mrtrix_parameters': {
                'wall_time': self.wall_time_step1,
                'output_verbosity': self.output_verbosity,
                'analysis_level': str(self.analysis_level),
                'analysis_level_config': self.analysis_level.parameters,
            },
            'fmriprep': self.step2_choice,
            'fmriprip_parameters': {
                'wall_time': self.wall_time_step2,
                'dof_6': self.dof_6,
                'mni_normalization': self.mni_normalization,
                'analysis_level': 'participant',
                'analysis_level_config': {
                    'skip_bids_validation': self.skip_bids,
                    'anat-only': self.anat_only,
                    'fs-no-reconall': self.no_reconall,
                }
            },
            'tvbconverter': self.step3_choice,
            'tvbconverter_parameters': {
                'wall_time': self.wall_time_step3
            }
        }

        with open(os.path.join(storage_path, self.PIPELINE_CONFIG_FILE), 'w') as f:
            json.dump(pipeline_config, f, indent=4)


KEY_PIPELINE = "ip-pipeline"


class PipelineStep1Form(Form):

    def __init__(self):
        super(PipelineStep1Form, self).__init__()
        self.wall_time_step1 = IntField(IPPipelineCreatorModel.wall_time_step1, name='wall_time_step1')
        self.output_verbosity = SelectField(IPPipelineCreatorModel.output_verbosity, name='output_verbosity')
        self.analysis_level = SelectField(EnumAttr(field_type=IPPipelineAnalysisLevelsEnum,
                                                   label="Select Analysis Level", required=True,
                                                   default=IPPipelineAnalysisLevelsEnum.PREPROC_LEVEL.instance,
                                                   doc="""Select the analysis level that the pipeline will be launched
                                                    on."""), name='analysis_level', subform=PreprocPipelineForm,
                                          session_key=KEY_PIPELINE)


class PipelineStep2Form(Form):

    def __init__(self):
        super(PipelineStep2Form, self).__init__()
        self.wall_time_step2 = IntField(IPPipelineCreatorModel.wall_time_step2, name='wall_time_step2')
        self.skip_bids = BoolField(IPPipelineCreatorModel.skip_bids)
        self.anat_only = BoolField(IPPipelineCreatorModel.anat_only)
        self.no_reconall = BoolField(IPPipelineCreatorModel.no_reconall)
        self.dof_6 = BoolField(IPPipelineCreatorModel.dof_6)
        self.mni_normalization = BoolField(IPPipelineCreatorModel.mni_normalization)


class PipelineStep3Form(Form):

    def __init__(self):
        super(PipelineStep3Form, self).__init__()
        self.wall_time_step3 = IntField(IPPipelineCreatorModel.wall_time_step3, name='wall_time_step3')


def zip_files(uploaded_files: List[cherrypy._cpreqbody.Part]) -> str:
    """
    Create a zip archive from a list of cherrypy Part objects
    zip the uploaded directory ( a list of Part objects ) by creating a temporary directory
    in which the directory tree is created then zip the created tree and return the path
    to the zip file created
    """
    zip_destination = TvbProfile.current.TVB_TEMP_FOLDER
    # create a temporary directory to write directory tree from which to create the archive
    temp_root_dir = tempfile.mkdtemp()
    # set the root which will be archived
    root_dir_name = uploaded_files[0].filename.split('/')[0]
    uploaded_root_dir = os.path.join(temp_root_dir, root_dir_name)
    pathlib.Path(uploaded_root_dir).mkdir(parents=True, exist_ok=True)

    # write the received files in the temporary dir
    for file in uploaded_files:
        # ignore macOS generated files
        if file.filename.endswith(IGNORED_FILES):
            continue

        path = os.path.join(temp_root_dir, os.path.normpath(file.filename))
        # create directory structure if it doesn't exist
        path_to_dir = os.path.dirname(path)
        pathlib.Path(path_to_dir).mkdir(parents=True, exist_ok=True)
        # write files
        with open(path, 'wb') as f:
            while True:
                data = file.file.read(8192)
                if not data:
                    break
                f.write(data)
    extension = 'zip'
    dir_zipped = shutil.make_archive(root_dir_name,
                                     extension,
                                     temp_root_dir)  # saves the archive on disk
    # move zip to TEMP dir and replace if a zip exists with that name
    zip_path = os.path.join(zip_destination, f'{root_dir_name}.{extension}')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.move(dir_zipped, zip_destination)

    # remove the temporary files as they are no longer needed
    shutil.rmtree(temp_root_dir)
    return zip_path


class IPPipelineCreatorForm(ABCAdapterForm):

    def __init__(self):
        super(IPPipelineCreatorForm, self).__init__()

        self.pipeline_job = SimpleLabelField("Pipeline Job1")
        self.mri_data = ValidatedTraitUploadField(IPPipelineCreatorModel.mri_data, '.zip', MRI_DATA_FIELD_NAME)
        self.participant_label = StrField(IPPipelineCreatorModel.participant_label)
        self.session_label = StrField(IPPipelineCreatorModel.session_label)
        self.task_label = StrField(IPPipelineCreatorModel.task_label)
        self.parcellation = SelectField(IPPipelineCreatorModel.parcellation, name='parcellation')
        self.nr_of_cpus = IntField(IPPipelineCreatorModel.nr_of_cpus, name='number_of_cpus')
        self.pipeline_steps_label = SimpleLabelField("Configure pipeline steps")

        self.step1_choice = BoolField(IPPipelineCreatorModel.step1_choice)
        self.step1_subform = FormField(PipelineStep1Form, 'pipeline_step1')

        self.step2_choice = BoolField(IPPipelineCreatorModel.step2_choice)
        self.step2_subform = FormField(PipelineStep2Form, 'pipeline_step2')

        self.step3_choice = BoolField(IPPipelineCreatorModel.step3_choice)
        self.step3_subform = FormField(PipelineStep3Form, 'pipeline_step3')

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
        self.step1_subform.form.analysis_level.data = type(analysis_level)
        self.step1_subform.form.analysis_level.subform_field = FormField(
            get_form_for_analysis_level(type(analysis_level)),
            'subform_analysis_level')
        self.step1_subform.form.fill_from_trait(trait)
        self.step2_subform.form.fill_from_trait(trait)
        self.step3_subform.form.fill_from_trait(trait)
        self.step1_subform.form.analysis_level.subform_field.form.fill_from_trait(analysis_level)

    def fill_trait(self, datatype):
        super(IPPipelineCreatorForm, self).fill_trait(datatype)
        if self.step1_subform.form.analysis_level.data == IPPipelineAnalysisLevelsEnum.PREPROC_LEVEL:
            datatype.analysis_level = PreprocAnalysisLevel()
        elif self.step1_subform.form.analysis_level.data == IPPipelineAnalysisLevelsEnum.PARTICIPANT_LEVEL:
            datatype.analysis_level = ParticipantAnalysisLevel()
        else:
            datatype.analysis_level = GroupAnalysisLevel()

        self.step1_subform.form.fill_trait(datatype)
        self.step2_subform.form.fill_trait(datatype)
        self.step3_subform.form.fill_trait(datatype)
        self.step1_subform.form.analysis_level.subform_field.form.fill_trait(datatype.analysis_level)

    def fill_from_post(self, form_data):
        # compress uploaded directory to zip
        form_data.update({MRI_DATA_FIELD_NAME: zip_files(form_data[MRI_DATA_FIELD_NAME])})

        super(IPPipelineCreatorForm, self).fill_from_post(form_data)
        self.step1_subform.form.analysis_level.subform_field.form = get_form_for_analysis_level(
            self.step1_subform.form.analysis_level.data.value)()
        self.step1_subform.form.analysis_level.subform_field.form.fill_from_post(form_data)


class IPPipelineCreator(ABCAdapter):
    _ui_name = "Launch Image Preprocessing Pipeline"
    _ui_description = "Launch Image Preprocessing Pipeline from tvb-web when it is deployed to EBRAINS"

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
        dest_path = os.path.join(storage_path, view_model.PIPELINE_DATASET_FILE)
        StorageInterface.copy_file(view_model.mri_data, dest_path)
        view_model.mri_data = dest_path
        h5.store_view_model(view_model, storage_path)

        view_model.to_json(storage_path)
