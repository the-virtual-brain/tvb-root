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
 * WARNING: This script is just adding some functionality specific to the model parameters on top of what is defined 
 * in /static/js/spatial/base_spatial.js. As such in all the cases when this script is used, you must first 
 * include base_spatial.js. In case you need to ADD FUNCTIONS here, either make sure you don't "overwrite" something
 * necessary from base_spatial.js, or just prefix your functions. (e.g. MP_SPATIAL_${function_name}).
 * ---------------------------------------===========================================--------------------------------------
 */

function MP_getSelectedParamName(){
    var maybeSelect = $("[name='_model_param']");
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
            $("#div_spatial_model_params").empty().append(data);
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
function changeEquationParamsForm(baseUrl, methodToCall, currentEquation, equationParamsDiv, fieldsWithEvents) {
    let url = baseUrl + "/" + methodToCall + "/" + currentEquation;
    $.ajax({
        url: url,
        type: 'POST',
        success: function (response) {
            var t = document.createRange().createContextualFragment(response);
            $("#" + equationParamsDiv).empty().append(t);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, equationParamsDiv]);
            setEventsOnFormFields(fieldsWithEvents, baseUrl, true);
            plotEquation(baseUrl)
        }
    })
}

function setModelParam(baseUrl, methodToCall, currentModelParam) {
    let url = baseUrl + "/" + methodToCall + "/" + currentModelParam;
    $.ajax({
        url: url,
        type: 'POST',
        success: function (data) {
            $("#div_spatial_model_params").empty().append(data);
            MP_displayFocalPoints();
        }
    })
}

function setParamAndRedrawChart(baseUrl, methodToCall, fieldName, fieldValue) {
    let currentParam = fieldName + '=' + fieldValue;
    let url = baseUrl + '/' + methodToCall + '?' + currentParam;
    $.ajax({
        url: url,
        type: 'POST',
        success: function () {
            plotEquation(baseUrl)
        }
    })
}

function prepareUrlParam(paramName, paramValue) {
    return paramName + '=' + paramValue;
}

function redrawPlotOnMinMaxChanges(baseUrl) {
    $('input[name="' + '_min_x' + '"]').change(function () {
        plotEquation(baseUrl, prepareUrlParam(this.name, this.value));
    });
    $('input[name="' + '_max_x' + '"]').change(function () {
        plotEquation(baseUrl, prepareUrlParam(this.name, this.value));
    });
}

function setEventsOnFormFields(fieldsWithEvents, url, onlyEquationParams = false) {
    let MODEL_PARAM_FIELD = 'set_model_parameter';
    let EQUATION_FIELD = 'set_equation';
    let EQUATION_PARAMS_FIELD = 'set_equation_param';

    if (onlyEquationParams === false) {
        $('select[name^="' + fieldsWithEvents[MODEL_PARAM_FIELD] + '"]').change(function () {
            setModelParam(url, MODEL_PARAM_FIELD, this.value)
        });

        let equationSelectFields = document.getElementsByName(fieldsWithEvents[EQUATION_FIELD]);
        for (let i=0; i<equationSelectFields.length; i++) {
            equationSelectFields[i].onclick = function () {
                changeEquationParamsForm(url, EQUATION_FIELD, this.value, 'equation_params',
                    fieldsWithEvents)
            };
        }
    }
    $('input[name^="' + fieldsWithEvents[EQUATION_PARAMS_FIELD] + '"]').change(function () {
        setParamAndRedrawChart(url, EQUATION_PARAMS_FIELD, this.name, this.value)
    });
}

function plotEquation(baseUrl, params=null) {
    let url = baseUrl + '/get_equation_chart';
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