/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
 *
 * This program is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this
 * program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * .. moduleauthor: Robert Vincze <robert.vincze@codemart.ro>
 **/


function importTumorDataset(projectId, algorithmId){
    displayMessage("Downloading Tumor Dataset from Ebrains has started." +
        " This operation is going to take a while, please wait...")
    doAjaxCall({
        type: "POST",
        url: "/project/launchloader/" + projectId + "/" + algorithmId,
        error: function(){
            displayMessage("We encountered an error while downloading the Tumor Dataset." +
                " Please try reload and then check the logs!", "errorMessage");
        }
    });
}
