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
 * Used for plotting equations.
 *
 * @param containerId the id of the container in which should be displayed the equations plot
 * @param url the url where should be made the call to obtain the html which contains the equations plot
 * @param formDataId the id of the form which contains the data for the equations
 * @param fieldsPrefixes a list with the prefixes of the fields at which the equation is sensible. If any
 * field that starts with one of this prefixes is changed than the equation chart will be redrawn.
 * @param axisDataId
 */
function BS_plotEquations(containerId, url, formDataId, fieldsPrefixes, axisDataId) {
    _plotEquations(containerId, url, formDataId, axisDataId);
    _applyEvents(containerId, url, formDataId, axisDataId, fieldsPrefixes);
}


/**
 * Private function.
 *
 * Updates the equations chart.
 */
function _plotEquations(containerId, url, formDataId, axisDataId) {
    var formInputs = $("#" + formDataId).serialize();
    var axisData = $('#' + axisDataId).serialize();
    doAjaxCall({
        async:false,
        type:'GET',
        url:url + "?" + formInputs + ';' + axisData,
        success:function (data) {
        	$("#" + containerId).empty().append(data);
        }
    });
}


/**
 * Private function.
 *
 * Applies change events on fields that starts with <code>fieldsPrefixes</code> to updated
 * the plotted equations. The fields should be on a form with id <code>formDataId</code>.
 */
function _applyEvents(containerId, url, formDataId, axisDataId, fieldsPrefixes) {
    for (var i = 0; i < fieldsPrefixes.length; i++) {
        $('select[name^="' + fieldsPrefixes[i] + '"]').change(function () {
            _plotEquations(containerId, url, formDataId, axisDataId);
        });
        $('input[name^="' + fieldsPrefixes[i] + '"]').change(function () {
            _plotEquations(containerId, url, formDataId, axisDataId);
        });
    }
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

