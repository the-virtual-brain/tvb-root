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
 **/

/*
 * ---------------------------------------===========================================--------------------------------------
 * WARNING: This script is just adding some functionality specific to the stimulus on top of what is defined 
 * in /static/js/spatial/base_spatial.js. As such in all the cases when this script is used, you must first 
 * include base_spatial.js. In case you need to ADD FUNCTIONS here, either make sure you don't "overwrite" something
 * necessary from base_spatial.js, or just prefix your functions. (e.g. STIM_SPATIAL_${function_name}).
 * ---------------------------------------===========================================--------------------------------------
 */

// the default weights values for a region stimulus; needed for reset action
var originalRegionStimulusWeights = [];
// the updated weights values for a region stimulus
var updatedRegionStimulusWeights = [];
var StimView;

function main(weights, selectionGID) {
    StimView = new TVBUI.RegionAssociatorView({
        selectionGID: selectionGID,
        onPut: _STIM_saveWeightForSelectedNodes,
        prepareSubmitData: function () {
            console.warn('This should not happen. Stimulus page does not use the submit logic of RegionAssociatorView. ');
            return false;
        }
    });
    originalRegionStimulusWeights = weights;
    updatedRegionStimulusWeights = originalRegionStimulusWeights.slice(0);
    GVAR_connectivityNodesWithPositiveWeight = [];
    StimView.setGridText(updatedRegionStimulusWeights);
}

function _STIM_server_update_scaling() {
    doAjaxCall({
        async: false,
        type: 'POST',
        data: {'scaling': JSON.stringify(updatedRegionStimulusWeights)},
        url: '/spatial/stimulus/region/update_scaling'
    });
}

/**
 * Saves the given weight for all the selected nodes.
 */
function _STIM_saveWeightForSelectedNodes() {
    const weightElement = $("#current_weight");
    let newWeight = parseFloat(weightElement.val());
    if (!isNaN(newWeight)) {
        for (let i = 0; i < GVAR_interestAreaNodeIndexes.length; i++) {
            const nodeIndex = GVAR_interestAreaNodeIndexes[i];
            updatedRegionStimulusWeights[nodeIndex] = newWeight;
        }
        weightElement.val("");
        _STIM_server_update_scaling();
    }
}


/**
 * Resets all the weights to their default values.
 */
function STIM_resetRegionStimulusWeights() {
    updatedRegionStimulusWeights = originalRegionStimulusWeights.slice(0);
    StimView.setGridText(updatedRegionStimulusWeights);
    _STIM_server_update_scaling();
}


/**
 * Submits all the region stimulus data to the server for creating a new Stimulus instance.
 *
 * @param actionUrl the url at which will be submitted the data.
 * @param nextStep
 * @param checkScaling
 */
function STIM_submitRegionStimulusData(actionUrl, nextStep, checkScaling) {
    if (checkScaling) {
        let scalingSet = false;
        for (let i = 0; i < updatedRegionStimulusWeights.length; i++) {
            if (updatedRegionStimulusWeights[i] !== 0) {
                scalingSet = true;
            }
        }
        if (!scalingSet) {
            displayMessage("You should set scaling that is not 0 for at least some nodes.", "warningMessage");
            return;
        }
    }
    _submitPageData(actionUrl, {'next_step': nextStep})
}


/**
 * *******************************************  *******************************************
 * CODE FOR SURFACE STIMULUS STARTING HERE.....
 * *******************************************  *******************************************
 */

/*
 * NOTE: The method is called through eval. Do not remove it.
 *
 * Parse the entry for the focal points and load them into js to be later used.
 */
function STIM_initFocalPoints(focal_points) {
    for (let i = 0; i < focal_points.length; i++) {
        const focalPoint = parseInt(focal_points[i]);
        BS_addedFocalPointsTriangles.push(focalPoint);
        BS_addedSurfaceFocalPoints.push(focalPoint);
        TRIANGLE_pickedIndex = focalPoint;
        BASE_PICK_moveBrainNavigator(true);
        BASE_PICK_addFocalPoint(focalPoint);
    }
    BS_drawSurfaceFocalPoints();
}


/*
 * Remove all previously defined focal points.
 */
function STIM_deleteAllFocalPoints() {
    for (let i = BS_addedFocalPointsTriangles.length - 1; i >= 0; i--) {
        BS_removeFocalPoint(i);
    }
}


/**
 * Collects the data needed for creating a SurfaceStimulus and submit it to the server.
 *
 * @param actionUrl the url at which will be submitted the data.
 * @param nextStep
 * @param includeFocalPoints
 */
function STIM_submitSurfaceStimulusData(actionUrl, nextStep, includeFocalPoints) {
    if (includeFocalPoints && BS_addedSurfaceFocalPoints.length < 1) {
        displayMessage("You should define at least one focal point.", "errorMessage");
        return;
    }
    let baseDict = {'next_step': nextStep};
    if (includeFocalPoints) {
        baseDict['defined_focal_points'] = JSON.stringify(BS_addedFocalPointsTriangles);
    }
    _submitPageData(actionUrl, baseDict)
}


/*
 * Gather the data from all the forms in the page and submit them to actionUrl.
 *
 * @param actionUrl: the url to which data will be submitted
 * @param baseData:
 */
function _submitPageData(actionUrl, baseData) {
    const pageForms = $('form');
    for (let i = 0; i < pageForms.length; i++) {
        pageForms[i].id = "form_id_" + i;
        const formData = getSubmitableData(pageForms[i].id, false);
        for (let key in formData) {
            baseData[key] = formData[key];
        }
    }

    const parametersForm = document.createElement("form");
    parametersForm.method = "POST";
    parametersForm.action = actionUrl;
    document.body.appendChild(parametersForm);

    for (let k in baseData) {
        const input = document.createElement('INPUT');
        input.type = 'hidden';
        input.name = k;
        input.value = baseData[k];
        parametersForm.appendChild(input);
    }

    parametersForm.submit();
}