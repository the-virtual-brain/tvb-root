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
 * ---------------------------------------=============================
 * This script defines functionality specific for Stimulus Region page
 * ---------------------------------------=============================
 */

// Following methods are used for handling events on dynamic forms
function setStimulusParamAndRedrawChart(methodToCall, fieldName, fieldValue) {
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
    $('#min_tmp_x').change(function () {
        plotEquation();
    });
    $('#max_tmp_x').change(function () {
        plotEquation();
    });
}

function setEventsOnStaticFormFields(fieldsWithEvents) {
    let CONNECTIVITY_FIELD = 'set_connectivity';
    let DISPLAY_NAME_FIELD = 'set_display_name';

    $('select[name^="' + fieldsWithEvents[CONNECTIVITY_FIELD] + '"]').change(function () {
        setStimulusParamAndRedrawChart(CONNECTIVITY_FIELD, this.name, this.value)
    });
    $('input[name^="' + fieldsWithEvents[DISPLAY_NAME_FIELD] + '"]').change(function () {
        setStimulusParamAndRedrawChart(DISPLAY_NAME_FIELD, this.name, this.value)
    });
}

function setEventsOnFormFields(fieldsWithEvents, div_id = 'temporal_params') {
    $('#' + div_id + ' input').change(function () {
        setStimulusParamAndRedrawChart('set_temporal_param', this.name, this.value)
    });
}

function prepareUrlParams() {
    min_field = $('#min_tmp_x')[0];
    min_params = prepareUrlParam(min_field.name, min_field.value);

    max_field = $('#max_tmp_x')[0];
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
