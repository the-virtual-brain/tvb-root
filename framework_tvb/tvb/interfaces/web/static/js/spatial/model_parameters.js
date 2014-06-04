/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and 
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
 *
 * This program is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU General Public License version 2 as published by the Free
 * Software Foundation. This program is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details. You should have received a copy of the GNU General 
 * Public License along with this program; if not, you can download it here
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0
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


/**
 * Draws sliders for the parameters of a model.
 *
 * @param paramSlidersData will be a dictionary of form:
 * dict = {'$param_name': { 'min': $min, 'max': $max, 'default': $default, 'name': '$param_name'},
 *          ...,
 *          'all_param_names': [list_with_all_parameters_names]
 *          }
 */
function MP_drawSlidersForModelParameters(paramSlidersData) {
    paramSlidersData = $.parseJSON(paramSlidersData);
    for (var i = 0; i < paramSlidersData['all_param_names'].length; i++) {
        var paramName = paramSlidersData['all_param_names'][i];
        var paramData = paramSlidersData[paramName];

        _drawSlider(paramName, paramData['min'], paramData['max'], paramData['step'], paramData['default']);
    }
}

function _drawSlider(name, minValue, maxValue, stepValue, defaultValue) {
    var sliderElement = $("#" + name);
    sliderElement.slider({
        value: defaultValue,
        min: minValue,
        max: maxValue,
        step: stepValue
    });

    $("#value_" + name).text(defaultValue);

    sliderElement.slider({
        change: function(event, ui) {
            if (GVAR_interestAreaNodeIndexes.length == 0) {
                displayMessage(NO_NODE_SELECTED_MSG, "errorMessage");
            } else {
	            var newValue = $('#' + name).slider("option", "value");
	            $("#value_" + name).text(newValue);
	            doAjaxCall({
	                async:false,
	                type:'GET',
	                url:'/spatial/modelparameters/regions/update_model_parameter_for_nodes/' + name + '/' + newValue + '/' + $.toJSON(GVAR_interestAreaNodeIndexes),
	                success:function (data) { }
	            });
			}
        }
    });
}


/**
 * @param parentDivId the id of the div in which are drawn the model parameters sliders
 */
function MP_resetParamSliders(parentDivId) {
    if (GVAR_interestAreaNodeIndexes.length > 0) {
        doAjaxCall({
            async:false,
            type:'GET',
            url:'/spatial/modelparameters/regions/reset_model_parameters_for_nodes/' + $.toJSON(GVAR_interestAreaNodeIndexes),
            success:function (data) {
                $("#" + parentDivId).empty().append(data);
            }
        });
    }
}

function _loadModelForConnectivityNode(connectivityNodeIndex, paramSlidersDivId) {
    if (connectivityNodeIndex >= 0) {
        doAjaxCall({
            async:false,
            type:'GET',
            url:'/spatial/modelparameters/regions/load_model_for_connectivity_node/' + connectivityNodeIndex,
            success:function (data) {
                $("#" + paramSlidersDivId).empty().append(data);
            }
        });
    }
}


/**
 * This method will toggle the selected node.
 * If there remains only one selected node than this method
 * will also load the model, for the remaining selected node,
 * into the phase plane viewer.
 *
 * @param nodeIndex the index of the node that hast to be toggled.
 */
function MP_toggleAndLoadModel(nodeIndex) {
    BS_toggleNodeSelection(nodeIndex);
    if (GFUNC_isNodeAddedToInterestArea(nodeIndex)) {
        if (GVAR_interestAreaNodeIndexes.length == 1) {
            _loadModelForConnectivityNode(GVAR_interestAreaNodeIndexes[0], 'div_spatial_model_params');
        } else if (GVAR_interestAreaNodeIndexes.length > 1) {
            _copyModel(GVAR_interestAreaNodeIndexes[0], [nodeIndex]);
        }
    }
}


function MP_copyAndLoadModel() {
    if (GVAR_interestAreaNodeIndexes.length != 0) {
        _copyModel(GVAR_interestAreaNodeIndexes[0], GVAR_interestAreaNodeIndexes.slice(1));
        _loadModelForConnectivityNode(GVAR_interestAreaNodeIndexes[0], 'div_spatial_model_params');
    }
}


/**
 * Replace the model of the nodes 'to_nodes' with the model of the node 'from_node'.
 *
 * @param fromNode the index of the node from where will be copied the model
 * @param toNodes a list with the nodes indexes for which will be replaced the model
 */
function _copyModel(fromNode, toNodes) {
    doAjaxCall({
        async:false,
        type:'POST',
        url:'/spatial/modelparameters/regions/copy_model/' + fromNode + '/' + $.toJSON(toNodes),
        success:function (data) {
        }
    });
}

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
            $("#div_spatial_model_params").empty().append(data);
            MP_displayFocalPoints();
        }
    });
}

function _MP_CallFocalPointsRPC(method, kwargs){
    var paramName = MP_getSelectedParamName();
    var url = '/spatial/modelparameters/surface/';
        url += method + '?model_param=' + paramName;
    for(var k in kwargs){
        if(kwargs.hasOwnProperty(k)) { url += '&' + k + '=' + kwargs[k]; }
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
// --------------------------------------------------------------------------------------
// ---------------------------- NOISE SPECIFIC SETTINGS ---------------------------------
// --------------------------------------------------------------------------------------

function NP_updateNoiseParameters(rootDivID) {
	var noiseValues = {};
	var displayedValue = '[';
	$('#' + rootDivID).find("input[id^='noisevalue']")
                      .each( function() {
                                    var nodeIdx = this.getAttribute('id').split('__')[1];
                                    noiseValues[parseInt(nodeIdx)] = $(this).val();
                                    displayedValue += $(this).val() + ' '
                                } );
	displayedValue = displayedValue.slice(0, -1);
	displayedValue += ']';
	var submitData = {'selectedNodes': $.toJSON(GVAR_interestAreaNodeIndexes),
					  'noiseValues': $.toJSON(noiseValues)};
					  
	doAjaxCall({  	type: "POST",
    			async: true,
				url: '/spatial/noiseconfiguration/update_noise_configuration',
				data: submitData, 
				traditional: true,
                success: function() {
                	var nodesLength = GVAR_interestAreaNodeIndexes.length;
				    for (var i = 0; i < nodesLength; i++) {
				        $("#nodeScale" + GVAR_interestAreaNodeIndexes[i]).text(displayedValue);
				        document.getElementById("nodeScale" + GVAR_interestAreaNodeIndexes[i]).className = "node-scale node-scale-selected"
				    }
				    GFUNC_removeAllMatrixFromInterestArea();
                }
            });
}

function _fill_node_selection_with_noise_values(data) {
    var nodesData = $.parseJSON(data);
    for (var i = 0; i < nodesData[0].length; i++) {
        var displayedValue = '[';
        for (var j = 0; j < nodesData.length; j++) {
            displayedValue += nodesData[j][i] + ' ';
        }
        displayedValue = displayedValue.slice(0, -1);
        displayedValue += ']';
        $("#nodeScale" + i).text(displayedValue);
    }
}

/*
 * Load the default values for the table-like connectivity node selection display.
 */
function NP_loadDefaultNoiseValues() {
	doAjaxCall({  	type: "POST",
    			async: true,
				url: '/spatial/noiseconfiguration/load_initial_values',
				traditional: true,
                success: function(r) {
                    _fill_node_selection_with_noise_values(r);
				    GFUNC_removeAllMatrixFromInterestArea();
                }
            });
}

function NP_toggleAndLoadNoise(nodeIndex) {
    BS_toggleNodeSelection(nodeIndex);
    if (GFUNC_isNodeAddedToInterestArea(nodeIndex)) {
        if (GVAR_interestAreaNodeIndexes.length == 1) {
            _loadNoiseValuesForConnectivityNode(GVAR_interestAreaNodeIndexes[0]);
        } else if (GVAR_interestAreaNodeIndexes.length > 1) {
            _copyNoiseConfig(GVAR_interestAreaNodeIndexes[0], [nodeIndex]);
        }
    }
}


function NP_copyAndLoadNoiseConfig() {
    if (GVAR_interestAreaNodeIndexes.length != 0) {
        _copyNoiseConfig(GVAR_interestAreaNodeIndexes[0], GVAR_interestAreaNodeIndexes.slice(1));
        _loadNoiseValuesForConnectivityNode(GVAR_interestAreaNodeIndexes[0]);
    }
}


function _loadNoiseValuesForConnectivityNode(connectivityNodeIndex) {
    if (connectivityNodeIndex >= 0) {
        doAjaxCall({
            async:false,
            type:'GET',
            url:'/spatial/noiseconfiguration/load_noise_values_for_connectivity_node/' + connectivityNodeIndex,
            showBlockerOverlay : true,
            success:function (data) {
            	var parsedData = $.parseJSON(data);
            	for (var key in parsedData) {
            		$('#noisevalue__' + key).val(parsedData[key]);
            	}
            }
        });
    }
}


/**
 * Replace the model of the nodes 'to_nodes' with the model of the node 'from_node'.
 *
 * @param fromNode the index of the node from where will be copied the model
 * @param toNodes a list with the nodes indexes for which will be replaced the model
 */
function _copyNoiseConfig(fromNode, toNodes) {
    doAjaxCall({
        async:false,
        type:'POST',
        url:'/spatial/noiseconfiguration/copy_configuration/' + fromNode + '/' + $.toJSON(toNodes),
        success: _fill_node_selection_with_noise_values
    });
}
