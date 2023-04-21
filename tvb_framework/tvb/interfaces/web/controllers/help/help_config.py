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
This module should contain all configurations necessary by help mechanism to
display correct online-help page for a given section/subsection/page.

.. moduleauthor:: Lia Domide  <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@cidemart.ro>
"""

from tvb.basic.logger.builder import get_logger
from tvb.interfaces.web.structure import WebStructure



class HelpConfig:
    """
    This class contains details that allows mapping of section + subsection pair 
    to correct online-help page/paragraph.
    """

    MAIN_HELP_PAGE = "UserGuide-UI.html"
    ANALYZE_HELP_PAGE = "UserGuide-UI_Analyze.html"
    BURST_HELP_PAGE = "UserGuide-UI_Simulator.html"
    CONNECTIVITY_HELP_PAGE = "UserGuide-UI_Connectivity.html"
    PROJECT_HELP_PAGE = "UserGuide-UI_Project.html"
    STIMULUS_HELP_PAGE = "UserGuide-UI_Stimulus.html"
    USER_HELP_PAGE = "UserGuide-UI_User.html"

    HELP_PAGE_PATH = "/statichelp/manuals/UserGuide"


    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self._mappings = {}
        self._load_mappings()


    def _load_mappings(self):
        """
        Here we map section + subsection to help page + paragraph
        """
        # Add mappings for root sections
        self._add_mapping(WebStructure.SECTION_USER, None, self.USER_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_PROJECT, None, self.PROJECT_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_BURST, None, self.BURST_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_ANALYZE, None, self.ANALYZE_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_STIMULUS, None, self.STIMULUS_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_CONNECTIVITY, None, self.CONNECTIVITY_HELP_PAGE, None)

        # Add mappings for USER subsections
        self._add_mapping(WebStructure.SECTION_USER, WebStructure.SUB_SECTION_LOGIN, self.USER_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_USER, WebStructure.SUB_SECTION_ACCOUNT, self.USER_HELP_PAGE, None)

        # Add mappings for PROJECT subsections
        self._add_mapping(WebStructure.SECTION_PROJECT, WebStructure.SUB_SECTION_OPERATIONS, self.PROJECT_HELP_PAGE, "operations")
        self._add_mapping(WebStructure.SECTION_PROJECT, WebStructure.SUB_SECTION_DATA_STRUCTURE, self.PROJECT_HELP_PAGE, "data-structure")
        self._add_mapping(WebStructure.SECTION_PROJECT, WebStructure.SUB_SECTION_LIST_PROJECTS, self.PROJECT_HELP_PAGE, "list-of-all-projects")
        self._add_mapping(WebStructure.SECTION_PROJECT, WebStructure.SUB_SECTION_PROPERTIES_PROJECT, self.PROJECT_HELP_PAGE, "basic-properties")
        self._add_mapping(WebStructure.SECTION_PROJECT, WebStructure.SUB_SECTION_FIGURES, self.PROJECT_HELP_PAGE, "image-archive")
        for subsection, paragraph in WebStructure.VISUALIZERS_ONLINE_HELP_SHORTCUTS.items():
            self._add_mapping(WebStructure.SECTION_PROJECT, subsection, self.PROJECT_HELP_PAGE, paragraph)

        # Add mappings for BURST subsections
        self._add_mapping(WebStructure.SECTION_BURST, WebStructure.SUB_SECTION_BURST, self.BURST_HELP_PAGE, None)
        self._add_mapping(WebStructure.SECTION_BURST, WebStructure.SUB_SECTION_MODEL_REGIONS, self.BURST_HELP_PAGE, "region-based-simulations")
        self._add_mapping(WebStructure.SECTION_BURST, WebStructure.SUB_SECTION_MODEL_SURFACE, self.BURST_HELP_PAGE, "surface-based-simulations")
        self._add_mapping(WebStructure.SECTION_BURST, WebStructure.SUB_SECTION_PHASE_PLANE, self.BURST_HELP_PAGE, "the-phase-plane-page")

        for subsection, paragraph in WebStructure.VISUALIZERS_ONLINE_HELP_SHORTCUTS.items():
            self._add_mapping(WebStructure.SECTION_BURST, subsection, self.BURST_HELP_PAGE, paragraph)

        # Add mappings for ANALYZE subsections
        for subsection, paragraph in WebStructure.ANALYZERS_ONLINE_HELP_SHORTCUTS.items():
            self._add_mapping(WebStructure.SECTION_ANALYZE, subsection, self.ANALYZE_HELP_PAGE, paragraph)

        # Add mappings for STIMULUS subsections
        self._add_mapping(WebStructure.SECTION_STIMULUS, WebStructure.SUB_SECTION_STIMULUS_SURFACE, self.STIMULUS_HELP_PAGE, "region-level-stimulus")
        self._add_mapping(WebStructure.SECTION_STIMULUS, WebStructure.SUB_SECTION_STIMULUS_REGION, self.STIMULUS_HELP_PAGE, "surface-level-stimulus")

        # Add mappings for CONNECTIVITY subsections
        self._add_mapping(WebStructure.SECTION_CONNECTIVITY, WebStructure.SUB_SECTION_CONNECTIVITY, self.CONNECTIVITY_HELP_PAGE, "long-range-connectivity")
        self._add_mapping(WebStructure.SECTION_CONNECTIVITY, WebStructure.SUB_SECTION_LOCAL_CONNECTIVITY, self.CONNECTIVITY_HELP_PAGE, "local-connectivity")
        self._add_mapping(WebStructure.SECTION_CONNECTIVITY, WebStructure.SUB_SECTION_ALLEN, self.CONNECTIVITY_HELP_PAGE, "allen-connectome-downloader")
        self._add_mapping(WebStructure.SECTION_CONNECTIVITY, WebStructure.SUB_SECTION_SIIBRA, self.CONNECTIVITY_HELP_PAGE, "siibra-connectivity-creator")


    def _add_mapping(self, section, subsection, page, paragraph):
        """
        This adds to main mapping structure an entry for:
        :param section: section for which to get help 
        :param subsection: subsection for which to get help (can be None)
        :param page: name of the help page corresponding to this section
        :param paragraph: id of the DIV / paragraph containing details about section/subsection (can be None
        """
        if section is None:
            raise Exception("Please provide section for which to add online-help.")
        if page is None:
            raise Exception("Please provide page where to find online-help.")

        key = self._generate_key(section, subsection)
        self._mappings[key] = (page + "#" + paragraph) if paragraph is not None else page


    def _generate_key(self, section, subsection):
        """
        Generates an unique key for section, subsection
        """
        return (section + "_" + subsection) if subsection is not None else section


    def get_help_url(self, section, subsection):
        """
        This method returns the URL for the online help page&paragraph corresponding
        to given section and subsection. 
        """
        if section is None:
            # If no section provided, show to user MAIN page.
            return self.HELP_PAGE_PATH + "/" + self.MAIN_HELP_PAGE

        key = self._generate_key(section, subsection)

        # Check if there is a mapping for this section/subsection combination.
        if key in self._mappings:
            return self.HELP_PAGE_PATH + "/" + self._mappings[key]
        else:
            self.logger.warning("There is no help mapping for section: %s and subsection: %s" % (section, subsection))

            # If no mapping found for pair section/subsection, try to find for section
            sect_key = self._generate_key(section, None)
            if sect_key in self._mappings:
                return self.HELP_PAGE_PATH + "/" + self._mappings[sect_key]
            else:
                self.logger.warning("and also there is no help mapping for section: %s." % section)
                # If not found, show to user MAIN page.
                return self.HELP_PAGE_PATH + "/" + self.MAIN_HELP_PAGE

