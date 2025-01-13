# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Adrian Ciu <adrian.ciu@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import smtplib
from email.mime.text import MIMEText
from tvb.basic.logger.builder import get_logger
from tvb.core.services.exceptions import EmailException

LOGGER = get_logger(__name__)


def send(address_from, address_to, email_subject, email_content, ignore_exception=True):
    """
    Sends an Email Message
    """
    try:
        email = MIMEText(email_content)
        email['From'] = address_from
        email['To'] = address_to
        email['Subject'] = email_subject

        server = smtplib.SMTP('mail.thevirtualbrain.org', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login('tvb_appserver', 'du5rEpratHAc')
        server.sendmail(address_from, address_to, email.as_string())
        server.quit()
        LOGGER.debug(f"Email sent to {address_to} with subject {email_subject}")
    except Exception as e:
        LOGGER.warn(f"Could not send email to {address_to}")
        if ignore_exception:
            return
        raise EmailException('Email could not be sent to user.', e)
