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

/*
 * ---------------------------------------===========================================--------------------------------------
 * WARNING: This script is just adding some functionality specific to the model parameters on top of what is defined 
 * in /static/js/spatial/base_spatial.js. As such in all the cases when this script is used, you must first 
 * include base_spatial.js. In case you need to ADD FUNCTIONS here, either make sure you don't "overwrite" something
 * necessary from base_spatial.js, or just prefix your functions. (e.g. MP_SPATIAL_${function_name}).
 * ---------------------------------------===========================================--------------------------------------
 */

function MP_getSelectedParamName(){
    var maybeSelect = $("[name='model_param']");
    if ( maybeSelect.prop('tagName') == "SELECT" ){
        return maybeSelect.val();
    }else{  // radio group
        return maybeSelect.filter(':checked').val();
    }
}

/**
 * Applies an equation for computing a model parameter.
 */
function MP_applyEquationForParameter() {
    var paramName = MP_getSelectedParamName();
    var formInputs = $("#form_spatial_model_param_equations").serialize();
    var plotAxisInputs = $('#equationPlotAxisParams').serialize();
    var url = '/spatial/modelparameters/surface/apply_equation?param_name=' + paramName + ';' + plotAxisInputs;
    url += '&' + formInputs;
    doAjaxCall({
        async:false,
        type:'POST',
        url:url,
        success:function (data) {
            const spatial_div = $("#div_spatial_model_params");
            renderWithMathjax(spatial_div, data, true);
            MP_displayFocalPoints();
        }
    });
}

function _MP_CallFocalPointsRPC(method, kwargs){
    var paramName = MP_getSelectedParamName();
    var url = '/spatial/modelparameters/surface/';
        url += method + '/' + paramName;
    for(var k in kwargs){
        if(kwargs.hasOwnProperty(k)) { url += '/' + kwargs[k]; }
    }
    doAjaxCall({
        async:false, type:'POST', url:url,
        success:function (data) {
            $("#focalPointsDiv").empty().append(data);
        }
    });
}

/**
 * Removes the given vertexIndex from the list of focal points specified for the
 * equation used for computing the selected model parameter.
 */
function MP_removeFocalPointForSurfaceModelParam(vertexIndex) {
    _MP_CallFocalPointsRPC('remove_focal_point', {'vertex_index': vertexIndex});
}


/**
 * Adds the selected vertex to the list of focal points specified for the
 * equation used for computing the selected model parameter.
 */
function MP_addFocalPointForSurfaceModelParam() {
    if (TRIANGLE_pickedIndex == undefined || TRIANGLE_pickedIndex < 0) {
        displayMessage(NO_VERTEX_SELECTED_MSG, "errorMessage");
        return;
    }
    _MP_CallFocalPointsRPC('apply_focal_point', {'triangle_index': TRIANGLE_pickedIndex});
}


/*
 * Redraw the left side (3D surface view) of the focal points recieved in the json.
 */
function MP_redrawSurfaceFocalPoints(focalPointsJson) {
    BASE_PICK_clearFocalPoints();
    BS_addedFocalPointsTriangles = [];
    var focalPointsTriangles = $.parseJSON(focalPointsJson);
    for (var i = 0; i < focalPointsTriangles.length; i++) {
        TRIANGLE_pickedIndex = parseInt(focalPointsTriangles[i]);
        BASE_PICK_moveBrainNavigator(true);
        BASE_PICK_addFocalPoint(TRIANGLE_pickedIndex);
        BS_addedFocalPointsTriangles.push(TRIANGLE_pickedIndex);
    }
}



/**
 * Displays all the selected focal points for the equation
 * used for computing selected model param.
 */
function MP_displayFocalPoints() {
    _MP_CallFocalPointsRPC('get_focal_points', {});
}

/**
 * Validates the surface model before submission
 * Currently only checks if there are focal points
 */
function MP_onSubmit(ev){
    // the client does not track the page state, it is in the server session
    // this is a heuristic to detect if there are any focal points
    var noFocalPoints = $("#focalPointsDiv li").length == 0;
    if (noFocalPoints){
        displayMessage('You have no focal points', 'errorMessage');
        ev.preventDefault();
    }
}

// Following methods are used for handling events on dynamic forms
function setModelParam(methodToCall, currentModelParam) {
    let url = refreshBaseUrl + "/" + methodToCall + "/" + currentModelParam;
    $.ajax({
        url: url,
        type: 'POST',
        success: function (data) {
            const spatial_model_div = $("#div_spatial_model_params");
            renderWithMathjax(spatial_model_div, data, true)
            MP_displayFocalPoints();
        }
    })
}

function setParamAndRedrawChart(methodToCall, fieldName, fieldValue) {
    let currentParam = fieldName + '=' + fieldValue;
    let url = refreshBaseUrl + '/' + methodToCall + '?' + currentParam;
    $.ajax({
        url: url,
        type: 'POST',
        success: function () {
            plotEquation()
        }
    })
}

function redrawPlotOnMinMaxChanges() {
    $('#min_x').change(function () {
        plotEquation();
    });
    $('#max_x').change(function () {
        plotEquation();
    });
}

function setEventsOnStaticFormFields(fieldsWithEvents) {
    let MODEL_PARAM_FIELD = 'set_model_parameter';

    $('select[name^="' + fieldsWithEvents[MODEL_PARAM_FIELD] + '"]').change(function () {
        setModelParam(MODEL_PARAM_FIELD, this.value)
    });
}

function setEventsOnFormFields(fieldsWithEvents, div_id) {
    $('#' + div_id + ' input').change(function () {
        setParamAndRedrawChart('set_equation_param', this.name, this.value)
    });
}

function prepareUrlParams() {
    min_field = $('#min_x')[0];
    min_params = prepareUrlParam(min_field.name, min_field.value);

    max_field = $('#max_x')[0];
    max_params = prepareUrlParam(max_field.name, max_field.value);

    params = min_params + '&' + max_params;
    return params;
}

function plotEquation(subformDiv = null) {
    let url = refreshBaseUrl + '/get_equation_chart';
    params = prepareUrlParams();
    if (params) {
        url += '?' + params
    }
    doAjaxCall({
        async: false,
        type: 'GET',
        url: url,
        success: function (data) {
            $("#" + 'equationDivId').empty().append(data);
        }
    });
}
