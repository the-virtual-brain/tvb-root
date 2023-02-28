/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need to download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

/* global variable to keep the base URL for each exploration page */
var refreshBaseUrl = '';

// This method is called by the template that includes the current script
function initialize_baseUrl(url = '') {
    refreshBaseUrl = url;
}


/* globals displayMessage, doAjaxCall */

// this array will contain all the focal points selected by the user
var BS_addedSurfaceFocalPoints = [];
// this array will contain all the triangles for the defined focal points
var BS_addedFocalPointsTriangles = [];

var NO_VERTEX_SELECTED_MSG = "You have no vertex selected.";
var NO_NODE_SELECTED_MSG = "You have no node selected.";


/**
 * Display the index for the selected vertex.
 */
function BS_displayIndexForThePickedVertex() {
    if (TRIANGLE_pickedIndex == undefined || TRIANGLE_pickedIndex < 0) {
        displayMessage("No triangle is selected.", "warningMessage");
    } else {
        displayMessage("The index of the selected triangle is " + TRIANGLE_pickedIndex, "infoMessage");
    }
}


/**
 * Generate the data from the currently configured stimulus/LC and display as movie.
 */
function BS_startSignalVisualization() {

    if (BS_addedSurfaceFocalPoints.length < 1) {
        displayMessage("You should define at least one focal point.", "errorMessage");
        return;
    }
    LEG_generateLegendBuffers();
    doAjaxCall({
        async:false,
        type:'GET',
        url:'/spatial/stimulus/surface/view_stimulus?focal_points=' + JSON.stringify(BS_addedFocalPointsTriangles),
        success:function (data) {
            data = $.parseJSON(data);
            if (data['status'] == 'ok') {
                STIM_PICK_setVisualizedData(data);
                $('.action-stop').removeClass('action-idle');
                $('.action-run').addClass('action-idle');
            } else {
                displayMessage(data['errorMsg'], "errorMessage");
            }
        },
        error: function(x) {
            if (x.status == 500) {
                displayMessage("An error occurred probably due to invalid parameters.", "errorMessage");
            }
        }
    });
}

function BS_stopSignalVisualization() {
	STIM_PICK_stopDataVisualization();
    $('.action-run').removeClass('action-idle');
    $('.action-stop').addClass('action-idle');
    LEG_legendBuffers = [];
}


/**
 * Displays all the focal points that were selected by the user.
 */
function BS_drawSurfaceFocalPoints() {
	var focalPointsContainer = $('.focal-points');
    focalPointsContainer.empty();
    var dummyDiv = document.createElement('DIV');
    for (var i = 0; i < BS_addedFocalPointsTriangles.length; i++) {
        dummyDiv.innerHTML = '<li> <a href="#" onclick="BS_centerNavigatorOnFocalPoint(' + BS_addedFocalPointsTriangles[i] + ')">Focal point ' + BS_addedFocalPointsTriangles[i] + '</a>' +
            '<a href="#" onclick="BS_removeFocalPoint(' + i + ')" title="Delete focal point: ' + BS_addedFocalPointsTriangles[i] + '" class="action action-delete">Delete</a>' +
            '</li>';
	    var focalPointElement = dummyDiv.firstChild;
	    focalPointsContainer.append(focalPointElement);
    }
}


/**
 * Move the brain navigator on this focal point.
 */
function BS_centerNavigatorOnFocalPoint(focalPointTriangle) {
	TRIANGLE_pickedIndex = parseInt(focalPointTriangle);
	BASE_PICK_moveBrainNavigator(true);
}


/**
 * Add a focal point for the currently selected surface.
 */
function BS_addSurfaceFocalPoint() {
    if (TRIANGLE_pickedIndex < 0) {
        displayMessage(NO_VERTEX_SELECTED_MSG, "errorMessage");
    } else if (BS_addedSurfaceFocalPoints.length >= 20) {
        displayMessage("The max number of focal points you are allowed to add is 20.", "errorMessage");
    } else {
        var valIndex = $.inArray(TRIANGLE_pickedIndex, BS_addedFocalPointsTriangles);
        if (valIndex < 0) {
        	displayMessage("Adding focal point with number: "+ (BS_addedSurfaceFocalPoints.length + 1)); //clear msg
            BS_addedSurfaceFocalPoints.push(VERTEX_pickedIndex);
            BS_addedFocalPointsTriangles.push(TRIANGLE_pickedIndex);
            BS_drawSurfaceFocalPoints();
            BASE_PICK_addFocalPoint(TRIANGLE_pickedIndex);
        } else {
        	displayMessage("The focal point " + TRIANGLE_pickedIndex + " is already in the focal points list.", "warningMessage"); //clear msg
        }
    }
}


/**
 * Removes a focal point.
 *
 * @param focalPointIndex the index of the focal point that has to be removed.
 */
function BS_removeFocalPoint(focalPointIndex) {
	var focalIndex = BS_addedFocalPointsTriangles[focalPointIndex];
	BASE_PICK_removeFocalPoint(focalIndex);
	BS_addedFocalPointsTriangles.splice(focalPointIndex, 1);
    BS_addedSurfaceFocalPoints.splice(focalPointIndex, 1);
    BS_drawSurfaceFocalPoints();
}

/**
 * Loads the interface found at the given url.
 */
function BS_loadEntity() {
    var selectedEntityGid = $("select[name='existentEntitiesSelect']").val();
    var myForm;

    if (selectedEntityGid == undefined || selectedEntityGid == "explicit-None-value" || selectedEntityGid.trim().length == 0) {
		myForm = document.getElementById("reset-to-default");
    } else {
		myForm = document.getElementById("load-existing-entity");
		$("#entity-gid").val(selectedEntityGid);
    }
    myForm.submit();
}
