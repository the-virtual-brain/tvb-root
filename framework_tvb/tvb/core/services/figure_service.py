# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Service layer, for storing/retrieving Resulting Figures in TVB.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Ciprian Tomoiaga <ciprian.tomoiaga@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
from PIL import Image
import base64
import xml.dom.minidom
from io import BytesIO
from tvb.basic.logger.builder import get_logger
from tvb.core import utils
from tvb.core.entities.file.data_encryption_handler import encryption_handler
from tvb.core.entities.model.model_operation import ResultFigure
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper


class FigureService:
    """
    Service layer for Figure entities.
    """
    _TYPE_PNG = "png"
    _TYPE_SVG = "svg"

    _BRANDING_BAR_PNG = os.path.join(os.path.dirname(__file__), "resources", "branding_bar.png")
    _BRANDING_BAR_SVG = os.path.join(os.path.dirname(__file__), "resources", "branding_bar.svg")

    _DEFAULT_SESSION_NAME = "Default"
    _DEFAULT_IMAGE_FILE_NAME = "snapshot."

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.file_helper = FilesHelper()

    def _write_png(self, store_path, export_data):
        img_data = base64.b64decode(export_data)                        # decode the image
        final_image = Image.open(BytesIO(img_data))                     # place it in a PIL stream

        branding_bar = Image.open(FigureService._BRANDING_BAR_PNG)      # place the branding bar over
        final_image.paste(branding_bar, (0, final_image.size[1] - branding_bar.size[1]), branding_bar)

        final_image.save(store_path)                                    # store to disk as PNG

    def _write_svg(self, store_path, export_data):
        dom = xml.dom.minidom.parseString(export_data)
        figureSvg = dom.getElementsByTagName('svg')[0]                  # get the original image

        dom = xml.dom.minidom.parse(FigureService._BRANDING_BAR_SVG)

        try:
            width = float(figureSvg.getAttribute('width').replace('px', ''))
            height = float(figureSvg.getAttribute('height').replace('px', ''))
        except ValueError:                                                      # defaults when dimensions are not given
            width = 1024
            height = 768
            figureSvg.setAttribute("width", str(width))
            figureSvg.setAttribute("height", str(height))

        finalSvg = dom.createElement('svg')                                     # prepare the final svg
        brandingSvg = dom.getElementsByTagName('svg')[0]                        # get the branding bar
        brandingSvg.setAttribute("y", str(height))                              # position it below the figure
        height += float(brandingSvg.getAttribute('height').replace('px', ''))   # increase original height with branding bar's height
        finalSvg.setAttribute("width", str(width))                              # same width as original figure
        finalSvg.setAttribute("height", str(height))

        finalSvg.appendChild(figureSvg)                                         # add the image
        finalSvg.appendChild(brandingSvg)                                       # and the branding bar

        # Generate path where to store image
        with open(store_path, 'w') as dest:
            finalSvg.writexml(dest)                                                 # store to disk

    def _image_path(self, project_name, img_type):
        "Generate path where to store image"
        images_folder = self.file_helper.get_images_folder(project_name)
        file_name = FigureService._DEFAULT_IMAGE_FILE_NAME + img_type
        return utils.get_unique_file_name(images_folder, file_name)

    @staticmethod
    def _generate_image_name(project, user, image_name):
        if not image_name:
            # default to a generic name prefix
            image_name = "figure"
        figure_count = dao.get_figure_count(project.id, user.id) + 1
        return 'TVB-%s-%s' % (image_name, figure_count)

    def store_result_figure(self, project, user, img_type, export_data, image_name=None):
        """
        Store into a file, Result Image and reference in DB.
        """
        store_path, file_name = self._image_path(project.name, img_type)
        image_name = self._generate_image_name(project, user, image_name)

        if img_type == FigureService._TYPE_PNG:            # PNG file from canvas
            self._write_png(store_path, export_data)
        elif img_type == FigureService._TYPE_SVG:          # SVG file from svg viewer
            self._write_svg(store_path, export_data)

        # Store entity into DB
        entity = ResultFigure(user.id, project.id, FigureService._DEFAULT_SESSION_NAME,
                              image_name, file_name, img_type)
        entity = dao.store_entity(entity)

        # Load instance from DB to have lazy fields loaded
        figure = dao.load_figure(entity.id)
        # Write image meta data to disk  
        self.file_helper.write_image_metadata(figure)
        encryption_handler.push_folder_to_sync(self.file_helper.get_project_folder(project))

    def retrieve_result_figures(self, project, user, selected_session_name='all_sessions'):
        """
        Retrieve from DB all the stored Displayer previews that belongs to the specified session. The
        previews are for current user and project; grouped by session.
        """
        result, previews_info = dao.get_previews(project.id, user.id, selected_session_name)
        for name in result:
            for figure in result[name]:
                figures_folder = self.file_helper.get_images_folder(project.name)
                figure_full_path = os.path.join(figures_folder, figure.file_path)
                # Compute the path 
                figure.file_path = utils.path2url_part(figure_full_path)
        return result, previews_info

    @staticmethod
    def load_figure(figure_id):
        """
        Loads a stored figure by its id.
        """
        return dao.load_figure(figure_id)

    def edit_result_figure(self, figure_id, **data):
        """
        Retrieve and edit a previously stored figure.
        """
        figure = dao.load_figure(figure_id)
        figure.session_name = data['session_name']
        figure.name = data['name']
        dao.store_entity(figure)

        # Load instance from DB to have lazy fields loaded.
        figure = dao.load_figure(figure_id)
        # Store figure meta data in an XML attached to the image.
        self.file_helper.write_image_metadata(figure)
        encryption_handler.push_folder_to_sync(self.file_helper.get_project_folder(figure.project.name))

    def remove_result_figure(self, figure_id):
        """
        Remove figure from DB and file storage.
        """
        figure = dao.load_figure(figure_id)

        # Delete all figure related files from disk.
        figures_folder = self.file_helper.get_images_folder(figure.project.name)
        path2figure = os.path.join(figures_folder, figure.file_path)
        if os.path.exists(path2figure):
            os.remove(path2figure)
            self.file_helper.remove_image_metadata(figure)
            encryption_handler.push_folder_to_sync(self.file_helper.get_project_folder(figure.project.name))
        # Remove figure reference from DB.
        result = dao.remove_entity(ResultFigure, figure_id)
        return result
