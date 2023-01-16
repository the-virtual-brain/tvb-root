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
Controller class for managing Figures saved in TVB.

.. moduleauthor:: Ciprian Tomoiaga <ciprian.tomoiaga@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import cherrypy
import formencode
from formencode import validators
from cherrypy.lib.static import serve_file
from tvb.core import utils
from tvb.core.services.figure_service import FigureService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.decorators import context_selected, check_user, handle_error
from tvb.interfaces.web.controllers.decorators import using_template, expose_page
from tvb.interfaces.web.controllers.project.project_controller import ProjectController
from tvb.storage.storage_interface import StorageInterface


@traced
class FigureController(ProjectController):
    """
    Resulting Figures are user-saved figures with specific visualizers or TVB pages which are considered important.
    """

    def __init__(self):
        ProjectController.__init__(self)
        self.storage_interface = StorageInterface()
        self.figure_service = FigureService()


    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    @context_selected
    def storeresultfigure(self, img_type, **kwargs):
        """Create preview for current displayed canvas and 
        store image in current session, for future comparison."""
        project = common.get_current_project()
        user = common.get_logged_user()
        suggested_name = kwargs.get("suggestedName")
        self.figure_service.store_result_figure(project, user, img_type, kwargs['export_data'], suggested_name)


    @expose_page
    @context_selected
    def displayresultfigures(self, selected_session='all_sessions'):
        """ Collect and display saved previews, grouped by session."""
        project = common.get_current_project()
        user = common.get_logged_user()
        data, all_sessions_info = self.figure_service.retrieve_result_figures(project, user, selected_session)
        manage_figure_title = "Figures for " + str(selected_session) + " category"
        if selected_session == 'all_sessions':
            manage_figure_title = "Figures for all categories"
        template_specification = dict(mainContent="project/figures_display", title="Stored Visualizer Previews",
                                      controlPage=None, displayControl=False, selected_sessions_data=data,
                                      all_sessions_info=all_sessions_info, selected_session=selected_session,
                                      manageFigureTitle=manage_figure_title)
        template_specification = self.fill_default_attributes(template_specification, subsection='figures')
        return template_specification


    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    @context_selected
    def editresultfigures(self, remove_figure=False, rename_session=False, remove_session=False, **data):
        """
        This method knows how to handle the following actions:
        remove figure, update figure, remove session and update session.
        """
        project = common.get_current_project()
        user = common.get_logged_user()

        redirect_url = '/project/figure/displayresultfigures'
        if "selected_session" in data and data["selected_session"] is not None and len(data["selected_session"]):
            redirect_url += '/' + data["selected_session"]
            del data["selected_session"]
        figure_id = None
        if "figure_id" in data:
            figure_id = data["figure_id"]
            del data["figure_id"]

        if cherrypy.request.method == 'POST' and rename_session:
            successfully_updated = True
            if "old_session_name" in data and "new_session_name" in data:
                figures_dict, _ = self.figure_service.retrieve_result_figures(project, user, data["old_session_name"])
                for _key, value in figures_dict.items():
                    for figure in value:
                        new_data = {"name": figure.name, "session_name": data["new_session_name"]}
                        success = self._update_figure(figure.id, **new_data)
                        if not success:
                            successfully_updated = False
                if successfully_updated:
                    common.set_info_message("The session was successfully updated!")
                else:
                    common.set_error_message("The session was not successfully updated! "
                                             "There could be some figures that still refer to the old session.")
        elif cherrypy.request.method == 'POST' and remove_session:
            successfully_removed = True
            if "old_session_name" in data:
                figures_dict, _ = self.figure_service.retrieve_result_figures(project, user, data["old_session_name"])
                for _key, value in figures_dict.items():
                    for figure in value:
                        success = self.figure_service.remove_result_figure(figure.id)
                        if not success:
                            successfully_removed = False
                if successfully_removed:
                    common.set_info_message("The session was removed successfully!")
                else:
                    common.set_error_message("The session was not entirely removed!")
        elif cherrypy.request.method == 'POST' and remove_figure and figure_id is not None:
            success = self.figure_service.remove_result_figure(figure_id)
            if success:
                common.set_info_message("Figure removed successfully!")
            else:
                common.set_error_message("Figure could not be removed!")
        elif figure_id is not None:
            self._update_figure(figure_id, **data)
        self.redirect(redirect_url)

    def _update_figure(self, figure_id, **data):
        """
        Updates the figure details to the given data.
        """
        try:
            data = EditPreview().to_python(data)
            self.figure_service.edit_result_figure(figure_id, **data)
            common.set_info_message('Figure details updated successfully.')
            return True
        except formencode.Invalid as excep:
            self.logger.debug(excep)
            common.set_error_message(excep.message)
            return False

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def downloadimage(self, figure_id):
        """
        Allow a user to download a figure.
        """
        figure = self.figure_service.load_figure(figure_id)
        image_folder = self.storage_interface.get_images_folder(figure.project.name)
        figure_path = os.path.join(image_folder, figure.file_path)
        return serve_file(figure_path, "image/" + figure.file_format, "attachment",
                          "%s.%s" % (figure.name, figure.file_format))

    @cherrypy.expose
    @handle_error(redirect=False)
    @using_template("overlay")
    @check_user
    def displayzoomedimage(self, figure_id):
        """
        Displays the image with the specified id in an overlay dialog.
        """
        figure = self.figure_service.load_figure(figure_id)
        figures_folder = self.storage_interface.get_images_folder(figure.project.name)
        figure_full_path = os.path.join(figures_folder, figure.file_path)
        figure_file_path = utils.path2url_part(figure_full_path)
        description = figure.session_name + " - " + figure.name
        template_dictionary = dict(figure_file_path=figure_file_path)
        return self.fill_overlay_attributes(template_dictionary, "Detail", description,
                                            "project/figure_zoom_overlay", "lightbox")


class EditPreview(formencode.Schema):
    """
    Validate edit action on Stored Preview 
    """
    name = formencode.All(validators.UnicodeString(not_empty=True))
    session_name = formencode.All(validators.UnicodeString(not_empty=True))
