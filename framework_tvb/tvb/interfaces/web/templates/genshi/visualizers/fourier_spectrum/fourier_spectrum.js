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


/**
 * Created by Dan Pop on 5/24/2017.
 */


var FourierSpectrum = {
    matrix_shape: null,
    data_matrix: null,
    x_values_array: null
};

function Fourier_fourier_spectrum_init(matrix_shape, plotName, xAxisName, yAxisName, x_min, x_max, url_base, svg_id) {

    Plot_plot1d_init(plotName, xAxisName, yAxisName, x_min, x_max, url_base, svg_id, Fourier_drawDataCurves);

    var x_values_array = [];
    for (var i = 0; i < matrix_shape[0]; i++) {
        x_values_array[i] = ((x_max - x_min) * i) / (matrix_shape[0] - 1) + x_min;
    }
    FourierSpectrum.matrix_shape = matrix_shape;
    FourierSpectrum.x_values_array = x_values_array;
}

function Fourier_drawDataCurves() {
    var svgContainer = Plot1d.svgContainer;
    var data_matrix = FourierSpectrum.data_matrix;
    var x_values_array = FourierSpectrum.x_values_array;
    var lineGen = Plot_drawDataCurves();

    for (var i = 0; i < data_matrix.length; i++) {
        var line_data = _mergeArrays(x_values_array, data_matrix[i]);
        svgContainer.append('svg:path')
            .attr('d', lineGen(line_data))
            .attr('stroke', "#469EEB")
            .attr('stroke-width', 1)
            .attr('fill', 'none');
    }
}

function _mergeArrays(array1, array2) {
    var result = [];
    for (var i = 0; i < array1.length; i++)
        result[i] = [array1[i], array2[i]];
    return result;
}

function Fourier_changeMode(mode) {
    Fourier_getData($("#state_select option:selected").val(), mode, $("#normalize_select option:selected").text());
}
function Fourier_changeState(state) {
    Fourier_getData(state, $("#mode_select option:selected").text(), $("#normalize_select option:selected").text());
}
function Fourier_changeNormalize(normalized) {
    Fourier_getData($("#state_select option:selected").val(), $("#mode_select option:selected").text(), normalized);
}

function Fourier_getData(state, mode, normalized) {
    let url_base = Plot1d.url_base;
    doAjaxCall({
        url: url_base + "selected_state=" + state + ";selected_mode=" + mode + ";normalized=" + normalized,
        type: 'POST',
        async: true,
        success: function (data) {
            data = $.parseJSON(data);
            FourierSpectrum.data_matrix = $.parseJSON(data.data_matrix);
            Plot1d.yMin = data.ymin;
            Plot1d.yMax = data.ymax;
            Plot1d.yAxisScale.domain([data.ymin, data.ymax]);
            Plot_drawGraph();
        }
    });
}