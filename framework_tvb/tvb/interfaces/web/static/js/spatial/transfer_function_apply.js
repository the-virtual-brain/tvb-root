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

function TF_applyTransferFunction(){
    doAjaxCall({
        async: false,
        type: 'POST',
        url: '/burst/tvb-o/apply_transfer_function',
        success: function (data){
            $("#" + 'histogramCanvasId').empty().append(data);
        }
    });
}

function TF_clearHistogram(){
    doAjaxCall({
        async: false,
        type: 'POST',
        url: '/burst/tvb-o/clear_histogram',
        success: function (data){
            $("#" + 'histogramCanvasId').empty().append(data);
        }
    });
}

function setTransferFunctionAndRedrawChart(methodToCall, fieldName, fieldValue) {
    let currentParam = fieldName + '=' + fieldValue;
    let url = refreshBaseUrl + '/' + methodToCall + '?' + currentParam;
    $.ajax({
        url: url,
        type: 'POST',
        success: function (data) {
            if (fieldName === 'connectivity_measure') {
                $("#" + 'histogramCanvasId').empty().append(data);
            } else {
                plotEquation();
            }
        }
    });
}

function plotEquation(subformDiv = null) {
    let url = refreshBaseUrl + '/get_equation_chart';
    doAjaxCall({
        async: false,
        type: 'GET',
        url: url,
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

function setEventsOnStaticFormFields(fieldsWithEvents){
    let CONNECTIVITY_MEASURE_FIELD = 'set_connectivity_measure';
    let MODEL_PARAM_FIELD = 'set_model_parameter';

    $('select[name^="' + fieldsWithEvents[CONNECTIVITY_MEASURE_FIELD] + '"]').change(function (){
        setTransferFunctionAndRedrawChart(CONNECTIVITY_MEASURE_FIELD, this.name, this.value)
    });

    $('select[name^="' + fieldsWithEvents[MODEL_PARAM_FIELD] + '"]').change(function (){
        setTransferFunctionAndRedrawChart(MODEL_PARAM_FIELD, this.name, this.value)
    });
}

function setEventsOnFormFields(elementType, div_id){
    $('#' + div_id + ' input').change(function () {
        setTransferFunctionAndRedrawChart('set_transfer_function_param', this.name, this.value)
    });
}