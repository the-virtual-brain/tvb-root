# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import os
import sys
import shutil
import cStringIO
from cfflib import load
from tempfile import gettempdir
from zipfile import ZipFile, ZIP_DEFLATED
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.adapters.uploaders.networkx_connectivity.parser import NetworkxParser
from tvb.adapters.uploaders.gifti.parser import GIFTIParser
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException, ParseException
from tvb.core.entities.storage import dao, transactional
from tvb.datatypes.connectivity import Connectivity
import tvb.datatypes.surfaces as surfaces




class CFF_Importer(ABCUploader):
    """
    Upload Connectivity Matrix from a CFF archive.
    """

    _ui_name = "CFF"
    _ui_subsection = "cff_importer"
    _ui_description = "Import from CFF archive one or multiple datatypes."
    logger = get_logger(__name__)


    def get_upload_input_tree(self):
        """
        Define as input parameter, a CFF archive.
        """
        tree = [{'name': 'cff', 'type': 'upload', 'required_type': '.cff',
                 'label': 'CFF archive', 'required': True,
                 'description': 'Connectome File Format archive expected'}]

        tree.extend(NetworkxParser.prepare_input_params_tree(prefix="CNetwork: "))

        tree.append({'name': 'should_center', 'type': 'bool', 'default': False, 'label': 'CSurface: Center surfaces',
                     'description': 'Center surfaces using vertices positions mean along axes'})
        return tree


    def get_output(self):
        return [Connectivity, surfaces.Surface]


    @transactional
    def launch(self, cff, should_center=False, **kwargs):
        """
        Process the uploaded CFF and convert read data into our internal DataTypes.
        :param cff: CFF uploaded file to process.
        """
        if cff is None:
            raise LaunchException("Please select CFF file which contains data to import")

        # !! CFF does logging by the means of `print` statements. We don't want these
        # logged to terminal as sys.stdout since we no longer have any control over them
        # so just buffer everything to a StringIO object and log them after operation is done.
        default_stdout = sys.stdout
        custom_stdout = cStringIO.StringIO()
        sys.stdout = custom_stdout

        try:
            conn_obj = load(cff)
            network = conn_obj.get_connectome_network()
            surfaces = conn_obj.get_connectome_surface()
            warning_message = ""
            results = []

            if network:
                partial = self._parse_connectome_network(network, warning_message, **kwargs)
                results.extend(partial)
            if surfaces:
                partial = self._parse_connectome_surfaces(surfaces, warning_message, should_center)
                results.extend(partial)

            self._cleanup_after_cfflib(conn_obj)

            current_op = dao.get_operation_by_id(self.operation_id)
            current_op.user_group = conn_obj.get_connectome_meta().title
            if warning_message:
                current_op.additional_info = warning_message
            dao.store_entity(current_op)

            return results

        finally:
            # Make sure to set sys.stdout back to it's default value so this won't
            # have any influence on the rest of TVB.
            print_output = custom_stdout.getvalue()
            sys.stdout = default_stdout
            custom_stdout.close()
            # Now log everything that cfflib2 outputes with `print` statements using TVB logging
            self.logger.debug("Output from cfflib library: %s" % print_output)


    def _parse_connectome_network(self, connectome_network, warning_message, **kwargs):
        """
        Parse data from a NetworkX object and save it in Connectivity DataTypes.
        """
        connectivities = []
        parser = NetworkxParser(self.storage_path, **kwargs)

        for net in connectome_network:
            try:
                net.load()
                connectivity = parser.parse(net.data)
                connectivity.user_tag_1 = str(connectivity.weights.shape[0]) + " regions"
                connectivities.append(connectivity)

            except ParseException:
                self.logger.exception("Could not process Connectivity")
                warning_message += "Problem when importing Connectivities!! \n"

        return connectivities


    def _parse_connectome_surfaces(self, connectome_surface, warning_message, should_center):
        """
        Parse data from a CSurface object and save it in our internal Surface DataTypes
        """
        surfaces, processed_files = [], []
        parser = GIFTIParser(self.storage_path, self.operation_id)

        for c_surface in connectome_surface:
            if c_surface.src in processed_files:
                continue

            try:
                # create a meaningful but unique temporary path to extract
                tmpdir = os.path.join(gettempdir(), c_surface.parent_cfile.get_unique_cff_name())
                self.log.debug("Extracting %s[%s] into %s ..." % (c_surface.src, c_surface.name, tmpdir))
                _zipfile = ZipFile(c_surface.parent_cfile.src, 'r', ZIP_DEFLATED)
                gifti_file_1 = _zipfile.extract(c_surface.src, tmpdir)

                gifti_file_2 = None
                surface_name, pair_surface = self._find_pair_file(c_surface, connectome_surface)
                if pair_surface:
                    self.log.debug("Extracting pair %s[%s] into %s ..." % (pair_surface.src, pair_surface.name, tmpdir))
                    gifti_file_2 = _zipfile.extract(pair_surface.src, tmpdir)

                surface_type = self._guess_surface_type(c_surface.src.lower())
                self.logger.info("We will import surface %s as type %s" % (c_surface.src, surface_type))
                surface = parser.parse(gifti_file_1, gifti_file_2, surface_type, should_center)
                surface.user_tag_1 = surface_name

                validation_result = surface.validate()
                if validation_result.warnings:
                    warning_message += validation_result.summary() + "\n"

                surfaces.append(surface)

                if pair_surface:
                    processed_files.append(pair_surface.src)
                processed_files.append(c_surface.src)

                if os.path.exists(tmpdir):
                    shutil.rmtree(tmpdir)

            except ParseException:
                self.logger.exception("Could not import a Surface entity.")
                warning_message += "Problem when importing Surfaces!! \n"
            except OSError:
                self.log.exception("Could not clean up temporary file(s).")

        return surfaces


    def _find_pair_file(self, current_surface, all_surfaces):
        """
        :param current_surface: CSurface instance
        :param all_surfaces: Iterable over CSurface objects
        :return: (string: surface_name based on the 1-2 pair files, CSurface: pair_surface or None)
        """
        surface_name = current_surface.name
        pair_surface = None
        pair_expected_name = self._is_hemisphere_file(current_surface.src)

        for srf in all_surfaces:
            if srf.src == pair_expected_name:
                pair_surface = srf
                surface_name = surface_name.replace('lh', '').replace('rh', '').replace('left', '').replace('right', '')
                break

        return surface_name, pair_surface


    def _is_hemisphere_file(self, file_name):
        """
        :param file_name: File Name to analyze
        :return: expected pair file name (replace left <--> right) or None is the file can not be parsed
        """
        if file_name:

            if file_name.count('lh') == 1:
                return file_name.replace('lh', 'rh')

            if file_name.count('left') == 1:
                return file_name.replace('left', 'right')

            if file_name.count('rh') == 1:
                return file_name.replace('rh', 'lh')

            if file_name.count('right') == 1:
                return file_name.replace('right', 'left')

        return None


    def _guess_surface_type(self, file_name):
        """
        Based on file_name, try to guess the surface type.
        e.g. when "pial" is found we guess Cortical Surface
        """
        guessed_type = surfaces.FACE

        if 'pial' in file_name or 'cortical' in file_name or 'cortex' in file_name:
            return surfaces.CORTICAL

        # TODO fill this guessing when we get more details
        return guessed_type


    def _cleanup_after_cfflib(self, conn_obj):
        """
        CFF doesn't delete temporary folders created, so we need to track and delete them manually!!
        """
        temp_files = []
        root_folder = gettempdir()

        for ele in conn_obj.get_all():
            if hasattr(ele, 'tmpsrc') and os.path.exists(ele.tmpsrc):
                full_path = ele.tmpsrc
                while os.path.split(full_path)[0] != root_folder and os.path.split(full_path)[0] != os.sep:
                    full_path = os.path.split(full_path)[0]
                #Get the root parent from the $gettempdir()$
                temp_files.append(full_path)

        conn_obj.close_all()
        conn_obj._zipfile.close()

        for ele in temp_files:
            try:
                if os.path.isdir(ele):
                    shutil.rmtree(ele)
                elif os.path.isfile(ele):
                    os.remove(ele)
            except OSError:
                self.logger.exception("Could not cleanup temporary files after import...")