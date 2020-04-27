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
 * ---------------------------------------=============================
 * This script defines functionality specific for Stimulus Region page
 * ---------------------------------------=============================
 */

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

function setStimulusParamAndRedrawChart(baseUrl, methodToCall, fieldName, fieldValue) {
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
    $('input[name="' + 'min_x' + '"]').change(function () {
        plotEquation(baseUrl, prepareUrlParam(this.name, this.value));
    });
    $('input[name="' + 'max_x' + '"]').change(function () {
        plotEquation(baseUrl, prepareUrlParam(this.name, this.value));
    });
}

function setEventsOnFormFields(fieldsWithEvents, url, onlyEquationParams = false) {
    let CONNECTIVITY_FIELD = 'set_connectivity';
    let TEMPORAL_FIELD = 'set_temporal';
    let DISPLAY_NAME_FIELD = 'set_display_name';
    let TEMPORAL_PARAMS_FIELD = 'set_temporal_param';

    if (onlyEquationParams === false) {
        $('select[name^="' + fieldsWithEvents[CONNECTIVITY_FIELD] + '"]').change(function () {
            setStimulusParamAndRedrawChart(url, CONNECTIVITY_FIELD, this.name, this.value)
        });
        $('input[name^="' + fieldsWithEvents[DISPLAY_NAME_FIELD] + '"]').change(function () {
            setStimulusParamAndRedrawChart(url, DISPLAY_NAME_FIELD, this.name, this.value)
        });

        //TODO: we want to have also support fields for this/ extract hardcoded strings
        let equationSelectFields = document.getElementsByName(fieldsWithEvents[TEMPORAL_FIELD]);
        for (let i=0; i<equationSelectFields.length; i++) {
            equationSelectFields[i].onclick = function () {
                changeEquationParamsForm(url, TEMPORAL_FIELD, this.value, 'temporal_params',
                    fieldsWithEvents)
            };
        }
    }
    $('input[name^="' + fieldsWithEvents[TEMPORAL_PARAMS_FIELD] + '"]').change(function () {
        setStimulusParamAndRedrawChart(url, TEMPORAL_PARAMS_FIELD, this.name, this.value)
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