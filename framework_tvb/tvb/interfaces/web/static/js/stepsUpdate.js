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


/**   When changing the selected value: hide/display correct sub-controlls div.
 *
 */
function updateDivContent(divID, component, parentDIV, radioComponent) {
    var selectedValue;
    if (radioComponent) {
        component = radioComponent;
        selectedValue = component.value;
    } else {
        selectedValue = $(component).val();
    }

    // todo: this is greedy. it selects descendant divs as well (rangers, dub-data types)
    // todo: it should be sufficient to select top level divs
    var sub_divs = $('div[id^="' + divID +'"]');
    sub_divs.hide();
    sub_divs.find('input').attr('disabled', 'disabled');
    sub_divs.find('select').attr('disabled', 'disabled');

    //Get the input fields that hold the active rangers
    var first_ranger = $('#range_1')[0];
    var second_ranger = $('#range_2')[0];

    if (first_ranger != null && second_ranger != null) {
        // call disable only on the active range components
        //Switched tabs so reset ranger values
        if (first_ranger.value.indexOf(component.name) === 0) {
            first_ranger.value = '0';
        }
        if (second_ranger.value.indexOf(component.name) === 0) {
            second_ranger.value = '0';
        }
        //RANGE_computeNrOfOps();
    }

    var sub_div = $('#' + divID + selectedValue);
    sub_div.show();

    //Get all the ranger type component from the div to be shown init to default values
    //then disable them.

    //Get all input type fields from the div to be shown that are not part of
    //a range component, remove the disable attribute and set display style to inline
    var inputs =  sub_div.find('input');
    inputs = inputs.filter(function() { return (this.id.indexOf("_RANGER") == -1 || this.id.indexOf('RANGER_buttonExpand') > 0); });
    inputs.removeAttr('disabled').css('display', 'inline');

    //Get all dictionary type fields from the div to be shown that are not part of
    //a range component, remove the disable attribute and set display style to inline
    var selector_name = 'div[id^="dict"]';
    var dicts;
    if (parentDIV && parentDIV.trim()) {
        dicts = $('#'+ parentDIV).find(selector_name);
    }else{
        dicts = $(selector_name);
    }
    dicts = dicts.filter(function() { return (this.id.indexOf("_RANGER") == -1); });
    dicts.removeAttr('disabled').css('display', 'inline');

    //Get all select type fields from the div to be shown that are not part of
    //a range component, remove the disable attribute and set display style to inline

    var selectors = sub_div.find('select').filter(function() { return (this.id.indexOf("_RANGER") == -1); });
    selectors.removeAttr('disabled').css('display', 'inline');

    // These triggers end up calling updateDivContent recursively
    sub_div.find('select').trigger("change");
    sub_div.find('input[type="radio"][checked="checked"]').trigger("change");
}


function updateDatatypeDiv(selectComponent) {
    //When updating a dataType, supplementary range checks are needed
    var component = eval(selectComponent);
    if (component.options.length == 0) {
        return;
    }

    var selectedName = component.options[component.selectedIndex].innerHTML;
    selectedName = selectedName.replace(/[^a-z]/gi,'');

    if (selectedName == 'All') {
        //First get the ranger hidden values to check for available spots
        var first_ranger = $('#range_1')[0];
        var second_ranger = $('#range_2')[0];

        if (first_ranger.value == '0') {
            //If a spot is available set it to the current component name and mark the previous selection as the new one
            first_ranger.value = component.name;
        }
        else if (second_ranger.value == '0') {
            second_ranger.value = component.name;
        } else {
            //If no selections are available, get the previous selection and re-set the select component
            component.selectedIndex = getPreviousSelection(component);
            displayMessage("TVB has reached the maximum number of supported parameters for Parameter Space Exploration!", 'warningMessage');
        }
    } else {
        setPreviousSelection(component, component.selectedIndex);
    }
}


function getPreviousSelection(component) {
    //  This will use the defaultSelected attribute to store not only the selected option but
    // also the one selected before, in case a ranger is no longer available.
    var options = component.options;
    for (var i = 0; i < options.length; i++) {
        if (options[i].defaultSelected == true) {
            return i;
        }
    }
    return undefined;
}

function setPreviousSelection(component, optionId) {
    //  This will set the defaultSelected in order to mark the previous selection, in case a return to it is needed.
    var options = component.options;
    for (var i=0; i < options.length; i++) {
        options[i].defaultSelected = false;
    }
    options[optionId].defaultSelected = true;
}


function multipleSelect(obj, divID) {
    //   When changing the selected values in a multi-select controll, update the sub-controlls div.
    var divs = $('div[id^="' + divID +'"]');
    divs.hide();
    divs.find('input').attr('disabled', 'disabled');
    divs.find('select').attr('disabled', 'disabled');
    for(var i=0; i<obj.length; i++) {
        if(obj[i].selected) {
            var div = $('#' + divID + obj[i].value);
            div.show();
            div.find('input').removeAttr('disabled');

            var selectRef = div.find('select');
            selectRef.removeAttr('disabled');
            selectRef.trigger("change");
            div.find('input[type="radio"][checked="checked"]').trigger("change");
        }
    }
}



function updateDimensionsSelect(selectName, parameters_prefix, required_dimension, expected_shape, operations, resetSession) {
    var divContentId = "dimensionsDiv_" + selectName;
    $("#" + divContentId).empty();
    var selectedOption = $("select[name='" + selectName + "'] option:selected");
    var gidValue = selectedOption.val();
    if (gidValue != undefined) {
        doAjaxCall({    async : false,
                    type: 'GET',
                    url: '/flow/gettemplatefordimensionselect/' + gidValue + "/" + selectName + "/" +
                            resetSession + "/" + parameters_prefix + "/" + required_dimension + "/" + expected_shape + "/" + operations,
                    success: function(data) {
                        $("#" + divContentId).append(data);
                    }
                });
        //if the parent select is disabled then disable all its children
        if ($("select[name='" + selectName + "']").is(':disabled')) {
            $("select[name^='" + selectName + "_" + parameters_prefix + "_']").each(function () {
                $(this).attr('disabled', 'disabled');
            });
        }
    }
}

/**
 * Updates the label which displays the shape of the resulted array
 * after the user has selected some entries
 */
function updateShapeLabel(selectNamePrefix, genshiParam, dimensionIndex) {
    var spanId = selectNamePrefix + "_span_shape";
    var hiddenShapeRef = $("#" + selectNamePrefix + "_array_shape");
    var hiddenShape = $.parseJSON(hiddenShapeRef.val());

    var expectedArrayDim = parseInt(($("#" + selectNamePrefix + "_expected_dim").val()).split("requiredDim_")[1]);
    var expectedSpanDimId = selectNamePrefix + "_span_expected_dim";

    var dimSelectId = "dimId_" + selectNamePrefix + "_" + genshiParam + "_" + dimensionIndex;
    var selectedOptionsCount = $("select[id='" + dimSelectId + "'] option:selected").size();
    var aggSelectId = "funcId_" + selectNamePrefix + "_" + genshiParam + "_" + dimensionIndex;
    var aggregationFunction = $("select[id='" + aggSelectId + "'] option:selected").val();
    aggregationFunction = aggregationFunction.split("func_")[1];

    if (aggregationFunction != "none") {
        hiddenShape[dimensionIndex] = 1;
    } else if (selectedOptionsCount == 0) {
        hiddenShape[dimensionIndex] = $("select[id='" + dimSelectId + "'] option").size();
    } else {
        hiddenShape[dimensionIndex] = selectedOptionsCount;
    }

    var dimension = 0;
    for (var i = 0; i < hiddenShape.length; i++) {
        if (hiddenShape[i] > 1) {
            dimension += 1;
        }
    }
    var dimLbl = "";
    if (dimension == 0) {
        dimLbl = " => 1 element";
    } else {
        dimLbl = " => " + dimension + "D array"
    }

    if (dimension != expectedArrayDim) {
        $("#" + expectedSpanDimId).text("Please select a " + expectedArrayDim + "D array");
    } else {
        $("#" + expectedSpanDimId).text("");
    }

    $("#" + spanId).text("Array shape:" + $.toJSON(hiddenShape) + dimLbl);
    hiddenShapeRef.val($.toJSON(hiddenShape));
}
