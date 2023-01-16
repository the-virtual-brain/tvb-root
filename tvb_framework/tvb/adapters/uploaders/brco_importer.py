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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.adapters.datatypes.db.annotation import ConnectivityAnnotationsIndex
from tvb.adapters.datatypes.h5.annotation_h5 import ConnectivityAnnotations
from tvb.adapters.uploaders.brco.parser import XMLParser
from tvb.core.adapters.abcuploader import ABCUploader, ABCUploaderForm
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import transactional
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitUploadField, TraitDataTypeSelectField
from tvb.core.neotraits.uploader_view_model import UploaderViewModel
from tvb.core.neotraits.view_model import Str, DataTypeGidAttr
from tvb.datatypes.connectivity import Connectivity


class BRCOImporterModel(UploaderViewModel):
    data_file = Str(
        label='Connectivity Annotations'
    )

    connectivity = DataTypeGidAttr(
        linked_datatype=Connectivity,
        label='Target Large Scale Connectivity',
        doc='The Connectivity for which these annotations were made'
    )


class BRCOImporterForm(ABCUploaderForm):

    def __init__(self):
        super(BRCOImporterForm, self).__init__()

        self.data_file = TraitUploadField(BRCOImporterModel.data_file, '.xml', 'data_file')
        self.connectivity = TraitDataTypeSelectField(BRCOImporterModel.connectivity, 'connectivity')

    @staticmethod
    def get_view_model():
        return BRCOImporterModel

    @staticmethod
    def get_upload_information():
        return {
            'data_file': '.xml'
        }


class BRCOImporter(ABCUploader):
    """
    Import connectivity data stored in the networkx gpickle format
    """
    _ui_name = "BRCO Ontology Annotations"
    _ui_subsection = "brco_importer"
    _ui_description = "Import connectivity annotations from BRCO Ontology"

    def get_form_class(self):
        return BRCOImporterForm

    def get_output(self):
        return [ConnectivityAnnotationsIndex]

    @transactional
    def launch(self, view_model):
        try:
            conn = self.load_traited_by_gid(view_model.connectivity)

            parser = XMLParser(view_model.data_file, conn.region_labels)
            annotations = parser.read_annotation_terms()

            result_ht = ConnectivityAnnotations()
            result_ht.set_annotations(annotations)
            result_ht.connectivity = conn

            result = self.store_complete(result_ht)
            return result
        except Exception as excep:
            self.log.exception("Could not process Connectivity Annotations")
            raise LaunchException(excep)
