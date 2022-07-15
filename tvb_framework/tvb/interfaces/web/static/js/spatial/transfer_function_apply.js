/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

function TF_submitAndRedraw(methodToCall, fieldName, fieldValue) {
    const postData = {};
    if (fieldName !== '') {
        postData[fieldName] = fieldValue;
    }
    let url = refreshBaseUrl + '/' + methodToCall;

    $.ajax({
        url: url,
        type: 'POST',
        data: postData,
        success: function (data) {
            if (['set_connectivity_measure', "apply_transfer_function", "clear_histogram"].includes(methodToCall)) {
                const result = $.parseJSON(data);
                const sect = $('section.col-2');
                const values = $.parseJSON(result['data']);
                const labels = $.parseJSON(result['labels']);
                const colors = $.parseJSON(result['colors']);
                redrawHistogram(sect.width(), sect.height(), values, labels, colors, result['xposition']);
            } else {
                plotEquation();
            }
        }
    });
}

function plotEquation(subformDiv = null) {
    let url = refreshBaseUrl + '/get_equation_chart';

    let min_x = 0;
    let max_x = 100;
    const min_x_input = document.getElementById('min_x');
    const max_x_input = document.getElementById('max_x');

    if (min_x_input) {
        min_x = min_x_input.value;
    }
    if (max_x_input) {
        max_x = max_x_input.value
    }
    doAjaxCall({
        async: false,
        type: 'POST',
        url: url,
        data: {'min_x': min_x, 'max_x': max_x},
        success: function (data) {
            $("#" + 'transferFunctionDivId').empty().append(data);
        }
    });
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
    let CONNECTIVITY_MEASURE_FIELD = 'set_connectivity_measure';
    let MODEL_PARAM_FIELD = 'set_model_parameter';

    $('select[name^="' + fieldsWithEvents[CONNECTIVITY_MEASURE_FIELD] + '"]').change(function () {
        TF_submitAndRedraw(CONNECTIVITY_MEASURE_FIELD, this.name, this.value)
    });

    $('select[name^="' + fieldsWithEvents[MODEL_PARAM_FIELD] + '"]').change(function () {
        TF_submitAndRedraw(MODEL_PARAM_FIELD, this.name, this.value)
    });
}

function setEventsOnFormFields(elementType, div_id) {
    $('#' + div_id + ' input').change(function () {
        TF_submitAndRedraw('set_transfer_function_param', this.name, this.value)
    });
}