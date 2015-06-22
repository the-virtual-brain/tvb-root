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
This module is responsible for generation of documentation (pdf, html, ...)
necessary for a distribution.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import shutil
import tempfile
from optparse import OptionParser
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.api_datatypes import DATATYPES_FOR_DOCUMENTATION
from tvb.analyzers.api_analyzers import ANALYZERS_FOR_DOCUMENTATION
from rst2pdf.createpdf import RstToPdf
from docutils.core import publish_file
import tvb_documentor.api_doc.generate_modules
from tvb_documentor.api_doc.generate_modules import process_sources, GenOptions



class DocGenerator:
    """
    This class will generate and copy in the distribution folder documents
    as PDF (into docs folder) and online help (in the web interface static folders).
    """

    ORIGINAL_CSS_STYLE = "html4css1.css"
    HTML_CSS_STYLE = "html_doc.css"
    PDF_STYLE = "pdf_doc.style"

    INSTALLATION_MANUAL_FOLDER = "InstallationManual"
    CONTRIBUTORS_MANUAL_FOLDER = "ContributorsManual"
    #SCIENTIFIC_REPORT_FOLDER = "ScientificReport"
    DEVELOPER_REFERENCE_FOLDER = "DeveloperReference"
    USER_GUIDE_FOLDER = "UserGuide"

    # Documents to be processed
    CONTRIBUTORS_MANUAL = {"name": "ContributorsManual", "folder": CONTRIBUTORS_MANUAL_FOLDER}
    INSTALL_MANUAL = {"name": "InstallationManual", "folder": INSTALLATION_MANUAL_FOLDER}
    #USER_SCIENTIFIC_REPORT = {"name": "UserScientificReport", "folder": SCIENTIFIC_REPORT_FOLDER}
    DEV_REFERENCE_MANUAL = {"name": "DeveloperReferenceManual", "folder": DEVELOPER_REFERENCE_FOLDER}
    USER_GUIDE = {"name": "UserGuide", "folder": USER_GUIDE_FOLDER}

    USER_GUIDE_UI = {"name": "UserGuide-UI", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_ANALYZE = {"name": "UserGuide-UI_Analyze", "folder": USER_GUIDE_FOLDER,
                             "extra_prefix": "\nTVB Analyzers \n................. \n\n",
                             "extra_docs_strings": ANALYZERS_FOR_DOCUMENTATION}
    USER_GUIDE_UI_BURST = {"name": "UserGuide-UI_Simulator", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_CONNECTIVITY = {"name": "UserGuide-UI_Connectivity", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_PROJECT = {"name": "UserGuide-UI_Project", "folder": USER_GUIDE_FOLDER,
                             "extra_prefix": "\nTVB Data Types \n................ \n\n",
                             "extra_docs_strings": DATATYPES_FOR_DOCUMENTATION,
                             "extra_sufix": "\n\n.. include:: UserGuide-UI_Simulator-Visualizers.rst \n\n"}
    USER_GUIDE_UI_STIMULUS = {"name": "UserGuide-UI_Stimulus", "folder": USER_GUIDE_FOLDER}
    USER_GUIDE_UI_USER = {"name": "UserGuide-UI_User", "folder": USER_GUIDE_FOLDER}

    # EXTENSIONS
    RST = ".rst"
    HTML = ".html"
    PDF = ".pdf"

    # Folders
    DOCS_SRC = "tvb_documentation"
    DOCS = "docs"
    API = "api"
    DIST_FOLDER = "dist"
    FW_FOLDER = "framework_tvb"
    MANUALS = "manuals"
    STYLES = "styles"
    ONLINE_HELP = "tvb/interfaces/web/static/help"
    SCREENSHOTHS = "screenshots"
    USERGUIDE_SCREENSHOTS = USER_GUIDE_FOLDER + "/" + SCREENSHOTHS
    ICONS_FOLDER = "icons"
    USERGUIDE_ICONS = USER_GUIDE_FOLDER + "/" + ICONS_FOLDER


    # Documents that will be transformed into PDF
    DOCS_TO_PDF = [USER_GUIDE, CONTRIBUTORS_MANUAL, DEV_REFERENCE_MANUAL]
                    # INSTALL_MANUAL,

    # Documents that will be transformed into HTML
    DOCS_TO_HTML = [USER_GUIDE_UI, USER_GUIDE_UI_ANALYZE, USER_GUIDE_UI_BURST,
                    USER_GUIDE_UI_CONNECTIVITY, USER_GUIDE_UI_PROJECT,
                    USER_GUIDE_UI_STIMULUS, USER_GUIDE_UI_USER]


    #paths relative to the tvb package that should not be documented
    EXCLUDES = ['simulator/demos', 'simulator/backend', 'simulator/plot',
                'interfaces/web/templates', 'adapters/portlets', 'tests']


    def __init__(self, tvb_root_folder, dist_folder, library_path):
        """
        Creates a new instance.
        :param tvb_root_folder: root tvb folder.  
        :param dist_folder: folder where distribution is built.  
        :param library_path: folder where TVB code is put into final distribution.  
        """
        self.logger = get_logger(self.__class__.__name__)
        self._dist_folder = dist_folder
        self._tvb_root_folder = tvb_root_folder
        self._manuals_folder = os.path.join(tvb_root_folder, self.DOCS_SRC, self.MANUALS)
        self._styles_folder = os.path.join(self._manuals_folder, self.STYLES)

        # Folders where to store results
        self._dist_docs_folder = os.path.join(self._dist_folder, self.DOCS)
        self._dist_api_folder = os.path.join(self._dist_folder, self.API)
        self._dist_online_help_folder = os.path.join(library_path, self.ONLINE_HELP)
        self._dist_styles_folder = os.path.join(self._dist_online_help_folder, self.STYLES)

        # Check if folders exist. If not create them 
        if not os.path.exists(self._dist_docs_folder):
            os.makedirs(self._dist_docs_folder)
        if os.path.exists(self._dist_online_help_folder):
            shutil.rmtree(self._dist_online_help_folder)
        if not os.path.exists(self._dist_api_folder):
            os.makedirs(self._dist_api_folder)
        os.makedirs(self._dist_online_help_folder)


    def generate_all_docs(self):
        """
        This method initiate creation of all TVB documents
        """
        self.logger.debug("Start generating TVB documentation")

        # Generate Online-Help content
        self.generate_online_help()

        # Convert all manuals to PDF
        self.generate_manuals_pdfs()

        # Generate API doc
        self.generate_api_doc()

        self.logger.debug("End generating TVB documentation")


    def generate_online_help(self):
        """
        This method generates from manuals RST files, HTML content used as online-help.
        """
        # Convert all necessary docs to HTML (online-help)
        for doc in self.DOCS_TO_HTML:
            source = os.path.join(self._manuals_folder, doc['folder'], doc['name'] + self.RST)

            concat_file = None
            try:
                # Generate a new source file which includes TVB constants
                concat_file = self._include_constants(source)
                self._include_extra(doc, concat_file)
                destination = os.path.join(self._dist_online_help_folder, doc['name'] + self.HTML)
                self._generate_html(concat_file, destination)
            finally:
                # Delete temporary file
                if concat_file is not None and os.path.exists(concat_file):
                    os.remove(concat_file)


        # We have to copy now necessary styles and images
        shutil.copytree(self._styles_folder, os.path.join(self._dist_online_help_folder, self.STYLES))
        shutil.copytree(os.path.join(self._manuals_folder, self.USERGUIDE_SCREENSHOTS),
                        os.path.join(self._dist_online_help_folder, self.SCREENSHOTHS))
        shutil.copytree(os.path.join(self._manuals_folder, self.USERGUIDE_ICONS),
                        os.path.join(self._dist_online_help_folder, self.ICONS_FOLDER))


    @staticmethod
    def _include_constants(source_file):
        """
        This method creates a new RST file which includes constants on the first
        line and rest of the original file.
        Returns a temporary file.
        """
        folder, source_file_name = os.path.split(source_file)
        new_file_path = os.path.join(folder, "tmp_" + source_file_name)

        with open(new_file_path, "w") as new_file:
            new_file.write(".. include :: ../templates/pdf_constants.rst \n")

            # Add the content of the original file to the new file
            with open(source_file, 'r') as original_file:
                data = original_file.read()
                new_file.write(data)

        return new_file_path


    @staticmethod
    def _include_extra(doc, source_file):
        """
        Append to the original RST file some extra Python Class documentation.
        """
        with open(source_file, "a") as new_file:
            if "extra_prefix" in doc:
                new_file.write(doc["extra_prefix"])
            if "extra_docs_strings" in doc:
                for class_name, doc_title in doc["extra_docs_strings"].iteritems():
                    new_file.write(doc_title + "\n")
                    new_file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n\n")
                    new_file.write(class_name.__doc__ + "\n\n")
            if "extra_sufix" in doc:
                new_file.write(doc["extra_sufix"])


    def _generate_html(self, rst_file_path, result_html_path):
        """
        Generates a HTML file based on the input RST file.
        """
        if not os.path.exists(rst_file_path):
            raise DocGenerateException("Provided RST file: %s does not exists." % rst_file_path)

        try:
            style_files = str(os.path.join(self._dist_styles_folder, self.ORIGINAL_CSS_STYLE) + "," +
                              os.path.join(self._dist_styles_folder, self.HTML_CSS_STYLE))
            overrides = {'stylesheet_path': style_files, 'date': False, 'time': False,
                         'math_output': 'MathJax', 'embed_stylesheet': False}

            publish_file(source_path=rst_file_path, destination_path=result_html_path,
                         writer_name="html", settings_overrides=overrides)
            self.logger.debug("HTML file %s generated successfully." % result_html_path)
        except Exception, ex:
            self.logger.exception("Could not generate HTML documentation")
            raise DocGenerateException("Could not generate HTML documentation", ex)


    def generate_manuals_pdfs(self):
        """
        This method generates PDF for all available manuals.
        """
        # Convert all DOCS into PDF
        for doc in self.DOCS_TO_PDF:
            source = os.path.join(self._manuals_folder, doc['folder'], doc['name'] + self.RST)
            self._generate_pdf(source, os.path.join(self._dist_docs_folder, doc['name'] + self.PDF))


    def _generate_pdf(self, rst_file_path, result_pdf_path):
        """
        Generates a PDF file based on the input RST file.
        """
        if not os.path.exists(rst_file_path):
            raise DocGenerateException("Provided RST file: %s does not exists." % rst_file_path)

        # We suppose all RST files have references relative to their folder
        # So we have to force rst2pdf to be executed in the folder where RST file resides. 
        basedir = os.path.split(rst_file_path)[0]
        baseurl = "file://" + basedir

        r2p = RstToPdf(style_path=[self._styles_folder], stylesheets=[self.PDF_STYLE],
                       basedir=basedir, baseurl=baseurl, breakside="any")
        try:
            with open(rst_file_path) as file_rst:
                content = file_rst.read()
            r2p.createPdf(text=content, source_path=rst_file_path, output=result_pdf_path)
            self.logger.debug("PDF file %s generated successfully." % result_pdf_path)
        except Exception, ex:
            self.logger.exception("Could not generate PDF documentation")
            raise DocGenerateException("Could not generate PDF documentation", ex)


    def generate_api_doc(self):
        """
        This generates API doc in HTML format and include this into distribution.
        """
        # Create a TMP folder where to store generated RST files.
        temp_folder = tempfile.mktemp(prefix='tvb_sphinx_rst_')

        try:
            os.makedirs(temp_folder)
            opts = GenOptions(None, 'rst', temp_folder, 'Project', 10, True, None)        

            import tvb
            # Start to create RST files needed for Sphinx to generate final HTML
            process_sources(opts, tvb.__path__, self.EXCLUDES)

            # Now generate HTML doc
            conf_folder = os.path.dirname(tvb_documentor.api_doc.generate_modules.__file__)
            args = ['anything',  # Ignored but must be there
                    '-b', 'html',  # Specify builder: html, dirhtml, singlehtml, txt, latex, pdf,
                    '-a',  # Use option "-a" : build all
                    '-q',  # Log only Warn and Error
                    '-c', conf_folder,  # Specify path where to find conf.py
                    '-d', temp_folder,
                    temp_folder,  # Source folder
                    self._dist_api_folder,  # Output folder
                    ]

            # This include should stay here, otherwise generation of PDFs will crash
            from sphinx.cmdline import main as sphinx_build

            sphinx_build(args)
        finally:
            # Delete temp folder
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)



class DocGenerateException(Exception):
    """
    Exception class for problem with generation of TVB documentation.
    """


    def __init__(self, message, parent_ex=None):
        Exception.__init__(self, message, parent_ex)
        self.message = message
        self.parent_ex = parent_ex



if __name__ == "__main__":
    #By default running this module we generate documentation
    CURRENT_FOLDER = os.getcwd()
    ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.join(CURRENT_FOLDER)))

    ABS_DIST_FOLDER = os.path.join(ROOT_FOLDER, DocGenerator.DIST_FOLDER)
    if os.path.exists(ABS_DIST_FOLDER):
        shutil.rmtree(ABS_DIST_FOLDER)
        os.makedirs(ABS_DIST_FOLDER)

    PARSER = OptionParser()
    PARSER.add_option("-a", "--api_doc_only", action="store_true", dest="api_doc_only", help="Generates only API DOC")
    PARSER.add_option("-p", "--pdf_only", action="store_true", dest="pdfs_only", help="Generates only manual PDFs")
    PARSER.add_option("-o", "--online_help_only", action="store_true",
                      dest="online_help_only", help="Generates only Online Help")
    (OPTIONS, ARGS) = PARSER.parse_args()

    GENERATOR = DocGenerator(ROOT_FOLDER, ABS_DIST_FOLDER, os.path.join(ROOT_FOLDER, DocGenerator.FW_FOLDER))

    # Check if partial things should be generated.
    if OPTIONS.api_doc_only or OPTIONS.pdfs_only or OPTIONS.online_help_only:

        if OPTIONS.api_doc_only:
            GENERATOR.generate_api_doc()

        if OPTIONS.pdfs_only:
            GENERATOR.generate_manuals_pdfs()

        if OPTIONS.online_help_only:
            GENERATOR.generate_online_help()

    else:
        GENERATOR.generate_all_docs()
    
