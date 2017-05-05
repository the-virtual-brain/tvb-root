/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and 
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

var RANGE_PARAMETER_1 = "range_1";
var RANGE_PARAMETER_2 = "range_2";
var SLIDER_SUFFIX = "_slider";
var STEP_INPUT_SUFFIX = "_stepInput";
var RANGE_LABELS_TD_SUFFIX = "_rangeLabelsTd";
var HIDDEN_SUFFIX = "_hidden";
var BUTTON_EXPAND_SUFFIX = "_buttonExpand";
var INPUT_FROM_RANGE = "_FromIdx";
var INPUT_TO_RANGE = "_ToIdx";

var ONE_HUNDRED = 1000;
var NORMALIZATION_VALUES = {};

function toggleRange(containerTableId, minValue, maxValue, stepValue, inputName){
    var first_ranger = document.getElementById(RANGE_PARAMETER_1);
    var second_ranger = document.getElementById(RANGE_PARAMETER_2);
    var notActive =  [first_ranger.value, second_ranger.value].indexOf(inputName) === -1;

    if (notActive){
        initRangeComponent(containerTableId, minValue, maxValue, stepValue, inputName);
    }else{
        disableRangeComponent(containerTableId, inputName);
    }
}

/**
 * The maxValue and the minValue will be normalized when we create the slider. All the time when we want to display the
 * slider values we should un-normalize them.
 *
 * The selected values (the start/end of the interval and the step) will be submitted in a field with the name equal to <code>inputName</code>
 *
 * @param containerTableId the id of the DIV in which will be displayed the component
 * @param minValue the first value of the range
 * @param maxValue the last value of the range
 * @param stepValue step for range
 * @param inputName Name of Input on which Range control is to be added
 */
function initRangeComponent(containerTableId, minValue, maxValue, stepValue, inputName) {
    var rangeComponentId = containerTableId + SLIDER_SUFFIX;
    var rangeLabelsTdId = containerTableId + RANGE_LABELS_TD_SUFFIX;

    var first_ranger = document.getElementById(RANGE_PARAMETER_1);
    var second_ranger = document.getElementById(RANGE_PARAMETER_2);
    if (first_ranger.value == '0'){
        first_ranger.value = inputName;
    } else if (second_ranger.value == '0'){
        second_ranger.value = inputName;
    } else {
        displayMessage("TVB has reached the maximum number of supported parameters for Parameter Space Exploration!", 'warningMessage');
        return;
    }

    NORMALIZATION_VALUES[containerTableId] = (1 / stepValue) * ONE_HUNDRED;
    $("input[name='"+ inputName + "']").hide().attr('disabled', 'disabled');
    document.getElementById(containerTableId).style.display = 'block';
    document.getElementById(rangeComponentId).style.display = 'block';

    $('#' + containerTableId + BUTTON_EXPAND_SUFFIX).val('Collapse Range');
    $('#' + containerTableId + HIDDEN_SUFFIX).removeAttr('disabled');
    $('#' + containerTableId + INPUT_FROM_RANGE).removeAttr('disabled');
    $('#' + containerTableId + INPUT_TO_RANGE).removeAttr('disabled');
    $('#' + containerTableId + STEP_INPUT_SUFFIX).removeAttr('disabled');

    $("#" + containerTableId).find('input[type=number]').on('blur', function(){updateRangeInterval(containerTableId);});

    //create the slider
    var rangerComponentRef = $("#" + rangeComponentId);
    var normalMin = _getNormalizedValue(minValue, containerTableId);
    var normalMax = _getNormalizedValue(maxValue, containerTableId);
    var normalStep = _getNormalizedValue(stepValue, containerTableId);
    rangerComponentRef.slider({
        range: true,
        min: normalMin,
        max: normalMax,
        step: normalStep,
        values: [normalMin, normalMax ],
        slide: function(event, ui) {
            _refreshFromToValues(ui.values[0], ui.values[1], containerTableId);
            _displayRangeLabels(minValue, maxValue, containerTableId, [ui.values[0], ui.values[1]], rangeLabelsTdId);
            _prepareDataForSubmit(containerTableId, ui.values);
        }
    }); 

    _refreshFromToValues(rangerComponentRef.slider("values", 0), rangerComponentRef.slider("values", 1), containerTableId);
    _displayRangeLabels(minValue, maxValue, containerTableId, [normalMin, normalMax], rangeLabelsTdId);

    var sliderValues = rangerComponentRef.slider("option", "values");
    _prepareDataForSubmit(containerTableId, sliderValues);
}


/*
 * Check that the input from a component is within min/max and that it's an actual number.
 * NOTE: This should be done by default by the input type="number" component but this is not
 * yet fully supported.
 */
function _validateInput(htmlComponent, minValue, maxValue) {
    var currentValue = parseFloat(htmlComponent.value);
    if (isNaN(currentValue)) {
        htmlComponent.value = currentValue;
    }
    if (currentValue < minValue) {
        htmlComponent.value = minValue;
    }
    if (currentValue > maxValue) {
        htmlComponent.value = maxValue;
    }
    return parseFloat(htmlComponent.value);
}

/*
 * Update the selected interval for the range contained in 'containerDivId'. If values are outside
 * of min/max values for this range, or if they are not valid floats, then fall back to min/man.
 */
function updateRangeInterval(containerDivId) {
    var fromComponent = document.getElementById(containerDivId + INPUT_FROM_RANGE);
    // mark
    var minMax = [parseFloat(fromComponent.min), parseFloat(fromComponent.max)];
    var fromValue = _validateInput(fromComponent, minMax[0], minMax[1]);

    var toComponent = document.getElementById(containerDivId + INPUT_TO_RANGE);
    var toValue = _validateInput(toComponent, fromValue, minMax[1]);

    var stepComponent = document.getElementById(containerDivId + STEP_INPUT_SUFFIX);
    var stepValue = _validateInput(stepComponent, 0, (toValue - fromValue));

    var normalizedFrom = _getNormalizedValue(fromValue, containerDivId);
    var normalizedTo = _getNormalizedValue(toValue, containerDivId);
    var normalizedStep = _getNormalizedValue(stepValue, containerDivId);

    var sliderDom = $('#' + containerDivId + SLIDER_SUFFIX);
    var rangeLabelsTdId = containerDivId + RANGE_LABELS_TD_SUFFIX;

    sliderDom.slider("option", "values", [normalizedFrom, normalizedTo]);
    _displayRangeLabels(minMax[0], minMax[1], containerDivId, [normalizedFrom, normalizedTo], rangeLabelsTdId);
    _prepareDataForSubmit(containerDivId, [normalizedFrom, normalizedTo]);
    sliderDom.slider("option", "step", normalizedStep);
}


/**
 * Using first_range/second_range values, make sure previously selected rangers are expanded again, and re-populated with values. 
 * @param {Object} rangeValues JSON ['first_range name', 'second_range field name']
 */
function updateRangeValues(rangeValues) {
    for (var i = 0; i < 2; i++){
        var rangeValue = rangeValues[i];
        if (rangeValue != '0') {
            var topDiv = $("table[id$='"+ rangeValue + "_RANGER']");

            if (!topDiv || ! topDiv[0]) {
                continue;
            }
            //todo: simplify these selectors
            var topContainerId = topDiv[0].id;
            $("input[id$='"+ rangeValue + "_RANGER_buttonExpand'][type='button']").click();
            $("input[name='"  + rangeValue + "'][type='text']").each(function () {
                var previous_json = $.parseJSON(this.value);
                var minValue = previous_json.minValue;
                var maxValue = previous_json.maxValue;
                var step = previous_json.step;
                var normalMin = _getNormalizedValue(minValue, topContainerId);
                var normalMax = _getNormalizedValue(maxValue, topContainerId);

                this.value = minValue;
                this.defaultValue = minValue;

                var rangeSliderRef = $("div[id$='"+ rangeValue + "_RANGER_slider']");
                rangeSliderRef.slider("option", "step", _getNormalizedValue(step, topContainerId));
                rangeSliderRef.slider("option", "values", [normalMin, normalMax]);

                var spinner_component = $("input[id$='"+ rangeValue + "_RANGER" + STEP_INPUT_SUFFIX + "']")[0];
                var fromInputField = $("input[id$='"+ rangeValue + "_RANGER" + INPUT_FROM_RANGE + "']")[0];
                var toInputField = $("input[id$='"+ rangeValue + "_RANGER" + INPUT_TO_RANGE + "']")[0];
                fromInputField.value = minValue;
                toInputField.value = maxValue;
                spinner_component.value = step;

                _displayRangeLabels(parseFloat(fromInputField.min), parseFloat(fromInputField.max),
                    topContainerId, [normalMin, normalMax], topContainerId + RANGE_LABELS_TD_SUFFIX);

                var sliderValues = $("#" + topContainerId + SLIDER_SUFFIX).slider("option", "values");
                _prepareDataForSubmit(topContainerId, sliderValues);
            });
        }
    }
    var first_ranger = document.getElementById(RANGE_PARAMETER_1);
    if (first_ranger != null) {
        //TODO: fix for chrome back problem. Should be removed after permanent fix.
        first_ranger.value = rangeValues[0];
        var second_ranger = document.getElementById(RANGE_PARAMETER_2);
        second_ranger.value = rangeValues[1];
    }
}


function _getDataSelectRangeComponent(containerDivId){
    //Get actual data from the checkboxes.
    var allOptions = $('div[id^="' + containerDivId + '"] input').filter(function() {
        return (this.id.indexOf("check") !== -1);
    });
    var data ={};
    for (var i=0; i<allOptions.length; ++i){
         data[allOptions[i].value] = allOptions[i].checked;
     }
     var hiddenFieldId = containerDivId + HIDDEN_SUFFIX;
     $("#" + hiddenFieldId).val($.toJSON(data));
}


function _getValues(minValue, maxValue, stepSpinnerId) {
    var stepSpinnerRef = $("#" + stepSpinnerId);
    var newStep = stepSpinnerRef.val();
    var values = [];
    if (newStep == undefined || newStep == 0) {
        return values;
    }

    if (newStep < 0) {
        newStep = 0;
    }
    if (newStep > (maxValue - minValue)) {
        newStep = (maxValue - minValue);
    }
    if ((maxValue - minValue) / newStep > 10){
        newStep = (maxValue - minValue) / 10;
    }
    stepSpinnerRef.value = newStep;
    var position = minValue;
    while(position < maxValue) {
        values.push(position.toFixed(2));
        position = position + parseFloat(newStep);
    }
    return values;
}

function _displayRangeLabels(minValue, maxValue, containerTableId, sliderValues, rangeLabelsTdId) {
    var stepSpinnerId = containerTableId + STEP_INPUT_SUFFIX;
    var rangeValues = _getValues(minValue, maxValue, stepSpinnerId);
    var row =   "<table width='100%'><tr>";
    var cellWidth = parseFloat(100 / rangeValues.length);

    for (var i = 0; i < rangeValues.length; i++) {
        row = row +  "<td style='width:" + cellWidth + "%;";
        if ((rangeValues[i] >= _getUnnormalizedValue(sliderValues[0], containerTableId)) &&
            (rangeValues[i] <= _getUnnormalizedValue(sliderValues[1], containerTableId))) {
            row = row + " background-color:orange;";
        }
        if (i === 0) {
            row = row + "' align='left'>" + rangeValues[i] + "</td>";
        } else {
            row = row + "' align='center'>" + rangeValues[i] + "</td>";
        }
    }
    row += "</tr></table>";

    $("#" + rangeLabelsTdId).empty().append(row);
}

function _prepareDataForSubmit(containerDivId, sliderValues) {
    var stepSpinnerId = containerDivId + STEP_INPUT_SUFFIX;
    var hiddenFieldId = containerDivId + HIDDEN_SUFFIX;

    var step = $("#" + stepSpinnerId).val();

    var data =  {minValue: _getUnnormalizedValue(sliderValues[0], containerDivId), maxValue: _getUnnormalizedValue(sliderValues[1], containerDivId), step: parseFloat(step)};
    $("#" + hiddenFieldId).val($.toJSON(data));
    RANGE_computeNrOfOps();
}


function _getUnnormalizedValue(number, containerRangeId) {
    var val = parseFloat(number / NORMALIZATION_VALUES[containerRangeId]);
    var nrDecimals = ('' + val).split('.');
    if (nrDecimals.length === 1) {
        return parseFloat(val.toFixed(2));
    }else {
        return parseFloat(val.toFixed(nrDecimals[1].length));
    }
}

function _getNormalizedValue(number, containerTableId) {
    return parseInt(number * NORMALIZATION_VALUES[containerTableId]);
}

function _refreshFromToValues(v0, v1, containerDivId) {
    document.getElementById(containerDivId + INPUT_FROM_RANGE).value = _getUnnormalizedValue(v0, containerDivId);
    document.getElementById(containerDivId + INPUT_TO_RANGE).value = _getUnnormalizedValue(v1, containerDivId);
}


function disableRangeComponent(containerTableId, inputName) {
    var first_ranger = document.getElementById(RANGE_PARAMETER_1);
    var second_ranger = document.getElementById(RANGE_PARAMETER_2);
    if (first_ranger.value == inputName){
        first_ranger.value = '0';
    } else if (second_ranger.value == inputName){
        second_ranger.value = '0';
    }

    var topTable = $('#' + containerTableId);
    topTable.hide();
    // Disable all input fields in the ranger
    topTable.find('input').attr('disabled', 'disabled');

    $("#" + inputName).removeAttr('disabled').css('display', 'block');
    $('#' + containerTableId + BUTTON_EXPAND_SUFFIX).val('Expand Range');
    $("#" + containerTableId).find('input[type=number]').off('blur');
    RANGE_computeNrOfOps();
}

/*************************************************************************************************************************
 * 			Functions that compute number of operations to be launched given the selected rangers
 *************************************************************************************************************************/


function _getOpsForRanger(rangerValue) {
    /*
     * Get the number of operations that will be generated by the given ranger values.
     */
    var maybeSelect = $('#' + rangerValue);
    if ( maybeSelect.prop('tagName') == "SELECT" ){
        return maybeSelect[0].options.length - 1;  // do not count the 'all' option
    }else{  // not a select; then we must have 3 text boxes : from to step
        var fromSelect = rangerValue + '_RANGER_FromIdx';
        var toSelect = rangerValue + '_RANGER_ToIdx';
        var stepValue = rangerValue + '_RANGER_stepInput';
        var fromVal = parseFloat($("[id$='" + fromSelect + "']").val());
        var toVal = parseFloat($("[id$='" + toSelect + "']").val());
        var stepVal = parseFloat($("[id$='" + stepValue + "']").val());
        var nrOps = Math.ceil((toVal - fromVal) / stepVal);
        if ((toVal - fromVal) % stepVal === 0) {
            nrOps += 1;
        }
        return nrOps;
    }
}

var THREASHOLD_WARNING = 500;
var THREASHOLD_ERROR = 50000;

function RANGE_computeNrOfOps() {
    /*
     * Compute the total number of operations that will be launched because of the ranges selected.
     */
    var first_ranger = document.getElementById(RANGE_PARAMETER_1);
    var second_ranger = document.getElementById(RANGE_PARAMETER_2);
    var nrOps = 1;
    if (first_ranger.value != '0') {
        nrOps = nrOps * _getOpsForRanger(first_ranger.value);
    }
    if (second_ranger.value != '0') {
        nrOps = nrOps * _getOpsForRanger(second_ranger.value);
    }
    var msg = "Range configuration: " + nrOps + " operations.";
    var className = "infoMessage";
    if (nrOps > THREASHOLD_WARNING) {
        className = "warningMessage";
    }
    if (nrOps > THREASHOLD_ERROR) {
        className = "errorMessage";
    }
    if (nrOps > 1) {
        // Unless greater than 1, it is not a range, so do not display a possible confusing message.
        displayMessage(msg, className);
    }
}

// this script is loaded dynamically. The comment below is a source map. This is to see the file in js tools
//@ sourceURL=range.js