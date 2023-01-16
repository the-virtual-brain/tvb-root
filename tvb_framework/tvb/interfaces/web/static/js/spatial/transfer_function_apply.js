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
                const values = $.parseJSON(result['data']);
                const labels = $.parseJSON(result['labels']);
                redrawHistogram(result['minColor'], result['maxColor'], values, labels, result['colors'], result['xposition']);
                if ("apply_transfer_function" == methodToCall) {
                    const divApplied = $("#appliedVectorsDivId");
                    divApplied.empty();
                    const applied = result['applied_transfer_functions'];
                    for (let it in applied) {
                        divApplied.append("<p><b> Parameter " + it + " : </b> " + applied[it] + "</p>");
                    }
                }
            } else {
                plotEquation();
            }
        }
    });
}

function plotEquation() {
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
            $("#transferFunctionDivId").empty().append(data);
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

function setEventsOnStaticFormFields() {

    $('select[name^="connectivity_measure"]').change(function () {
        TF_submitAndRedraw('set_connectivity_measure', this.name, this.value)
    });

    $('select[name^="model_param"]').change(function () {
        TF_submitAndRedraw('set_model_parameter', this.name, this.value)
    });
}

function setEventsOnFormFields(_elementType, div_id) {
    $('#' + div_id + ' input').change(function () {
        TF_submitAndRedraw('set_transfer_function_param', this.name, this.value)
    });
}