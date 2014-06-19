# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
from nibabel.gifti import giftiio
from tvb.adapters.uploaders.handler_connectivity import networkx2connectivity, KEY_NOSE
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao, transactional
import tvb.adapters.uploaders.handler_surface as handler_surface
import tvb.adapters.uploaders.constants as ct




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
        return [{'name': 'cff', 'type': 'upload', 'required_type': '.cff',
                 'label': 'CFF archive', 'required': True,
                 'description': 'Connectome File Format archive expected, with GraphML, Timeseries or GIFTI inside.'}]


    def get_output(self):
        return []


    def _prelaunch(self, operation, uid=None, available_disk_space=0, **kwargs):
        """
        Overwrite method in order to return the correct number of stored dataTypes.
        """
        self.nr_of_datatypes = 0
        msg, _ = ABCUploader._prelaunch(self, operation, uid=None, **kwargs)
        return msg, self.nr_of_datatypes


    @transactional
    def launch(self, cff):
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
            cdatas = conn_obj.get_connectome_data()

            warning_message = ""

            if network:
                try:
                    self._parse_connectome_network(network)
                except Exception, excep:
                    self.log.exception(excep)
                    warning_message += "Problem when importing a Connectivity!! \n"

            if surfaces:
                try:
                    self._parse_connectome_surfaces(surfaces, cdatas)
                except Exception, excep:
                    self.log.exception(excep)
                    warning_message += "Problem when importing Surface (or related attributes: LocalConnectivity/RegionMapping) !! \n"

            self._cleanup_after_cfflib(conn_obj)

            current_op = dao.get_operation_by_id(self.operation_id)
            current_op.user_group = conn_obj.get_connectome_meta().title
            if len(warning_message) > 0:
                current_op.additional_info = warning_message
            dao.store_entity(current_op)
        finally:
            # Make sure to set sys.stdout back to it's default value so this won't
            # have any influence on the rest of TVB.
            print_output = custom_stdout.getvalue()
            sys.stdout = default_stdout
            custom_stdout.close()
            # Now log everything that cfflib2 outputes with `print` statements using TVB logging
            self.logger.info("Output from cfflib2 library: %s" % (print_output,))


    def _parse_connectome_network(self, connectome_network):
        """
        Parse data from a NetworkX object and save it in Connectivity DataTypes.
        """
        for net in connectome_network:
            net.load()
            meta = net.get_metadata_as_dict()
            conn = networkx2connectivity(net.data, self.storage_path, meta.get(KEY_NOSE))
            self.nr_of_datatypes += 1
            self._capture_operation_results([conn], meta.get(ct.KEY_UID))


    def _parse_connectome_surfaces(self, connectome_surface, connectome_data):
        """
        Parse data from a CSurface object and save it in our internal Surface DataTypes
        """
        for c_surface in connectome_surface:
            # create a meaningful but unique temporary path to extract
            tmpdir = os.path.join(gettempdir(), c_surface.parent_cfile.get_unique_cff_name())
            self.log.debug("Using temporary folder for CFF import:" + tmpdir)
            _zipfile = ZipFile(c_surface.parent_cfile.src, 'r', ZIP_DEFLATED)
            gifti_file = _zipfile.extract(c_surface.src, tmpdir)
            gifti_img = giftiio.read(gifti_file)
            surface_meta = gifti_img.meta.get_metadata()
            res, uid = None, None

            if surface_meta.get(ct.SURFACE_CLASS) == ct.CLASS_SURFACE:
                vertices, normals, triangles = None, None, None
                for one_data in connectome_data:
                    cd_meta = one_data.get_metadata_as_dict()
                    if ct.KEY_UID in cd_meta and surface_meta[ct.KEY_UID] == cd_meta[ct.KEY_UID]:
                        if cd_meta[ct.KEY_ROLE] == ct.ROLE_VERTICES:
                            vertices = one_data
                        if cd_meta[ct.KEY_ROLE] == ct.ROLE_NORMALS:
                            normals = one_data
                        if cd_meta[ct.KEY_ROLE] == ct.ROLE_TRIANGLES:
                            triangles = one_data
                res, uid = handler_surface.gifti2surface(gifti_img, vertices, normals,
                                                         triangles, self.storage_path)
                self.nr_of_datatypes += 1
                self._capture_operation_results([res], uid)

            elif surface_meta.get(ct.SURFACE_CLASS) == ct.CLASS_CORTEX:
                for one_data in connectome_data:
                    cd_meta = one_data.get_metadata_as_dict()
                    if ct.KEY_UID in cd_meta and surface_meta[ct.KEY_UID] == cd_meta[ct.KEY_UID]:
                        if cd_meta[ct.KEY_ROLE] == ct.ROLE_REGION_MAP:
                            res, uid = handler_surface.cdata2region_mapping(one_data, surface_meta, self.storage_path)
                        if cd_meta[ct.KEY_ROLE] == ct.ROLE_LOCAL_CON:
                            res, uid = handler_surface.cdata2local_connectivity(one_data, surface_meta,
                                                                                self.storage_path)
                        if res is not None:
                            self.nr_of_datatypes += 1
                            self._capture_operation_results([res], uid)
            try:
                if os.path.exists(tmpdir):
                    shutil.rmtree(tmpdir)
            except OSError:
                self.log.exception("could not clean up temp file")


    def _cleanup_after_cfflib(self, conn_obj):
        """
        !! CFF doesn't delete temporary folders created,
           so we need to track and delete them manually!!
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