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


/**
 * Created by Dan Pop on 5/24/2017.
 */


var ComplexCoherence = {
    AVAILABLE_COLORS: [{hex_color: '#0F94DB', hex_face_color: '#469EEB'},
        {hex_color: '#16C4B9', hex_face_color: '#0CF0E1'},
        {hex_color: '#CC4F1B', hex_face_color: '#FF9848'}],
    cohAvDataCurve: null,
    cohAreaDataCurve: null,
    available_spectrum: null,
    hex_color: null,
    hex_face_color: null
};

function Complex_complex_coherence_init(plotName, xAxisName, yAxisName, x_min, x_max, url_base, available_spectrum, svg_id) {

    Plot_plot1d_init(plotName, xAxisName, yAxisName, x_min, x_max, url_base, svg_id, Complex_drawDataCurves);

    ComplexCoherence.available_spectrum = $.parseJSON(available_spectrum);
    ComplexCoherence.hex_color = '#0F94DB';
    ComplexCoherence.hex_face_color = '#469EEB';
}

function _Complex_loadData(coh_spec_sd, coh_spec_av) {
    var x_min = Plot1d.xMin;
    var x_max = Plot1d.xMax;
    var cohSdDataY = $.parseJSON(coh_spec_sd);
    var cohSdDataX = [];
    var cohAreaDataCurve = [];
    var cohAvDataY = $.parseJSON(coh_spec_av);
    var cohAvDataCurve = [];
    for (var i = 0; i < cohSdDataY.length; i++) {
        cohSdDataX[i] = ((x_max - x_min) * i) / (cohSdDataY.length - 1) + x_min;
        cohAvDataCurve[i] = [cohSdDataX[i], cohAvDataY[i]];
        cohAreaDataCurve[i] = [cohSdDataX[i], cohAvDataY[i] - cohSdDataY[i], cohAvDataY[i] + cohSdDataY[i]];
    }
    ComplexCoherence.cohAvDataCurve = cohAvDataCurve;
    ComplexCoherence.cohAreaDataCurve = cohAreaDataCurve;
}

function Complex_drawDataCurves() {
    var cohAvDataCurve = ComplexCoherence.cohAvDataCurve;
    var cohAreaDataCurve = ComplexCoherence.cohAreaDataCurve;
    var svgContainer = Plot1d.svgContainer;
    var xAxisScale = Plot1d.xAxisScale;
    var yAxisScale = Plot1d.yAxisScale;
    var area = d3.svg.area()
        .x(function (d) {
            return xAxisScale(d[0]);
        })
        .y0(function (d) {
            return yAxisScale(d[1]);
        })
        .y1(function (d) {
            return yAxisScale(d[2]);
        });

    var lineGen = Plot_drawDataCurves();

    svgContainer.append('svg:path')
        .datum(cohAreaDataCurve)
        .attr("fill", ComplexCoherence.hex_face_color)
        .attr("stroke-width", 0)
        .attr("d", area);

    svgContainer.append('svg:path')
        .attr('d', lineGen(cohAvDataCurve))
        .attr('stroke', ComplexCoherence.hex_color)
        .attr('stroke-width', 2)
        .attr('fill', 'none');
}

function Complex_getSpectrum(spectrum) {
    let url_base = Plot1d.url_base;
    doAjaxCall({
        url: url_base + "selected_spectrum=" + spectrum,
        type: 'POST',
        async: true,
        success: function (data) {
            data = $.parseJSON(data);
            _Complex_loadData(data.coh_spec_sd, data.coh_spec_av);
            Plot1d.yMin = data.ymin;
            Plot1d.yMax = data.ymax;
            Plot1d.yAxisScale.domain([data.ymin, data.ymax]);
            _Complex_updateColourForSpectrum(spectrum);
            Plot_drawGraph();
        }
    });
}

function _Complex_updateColourForSpectrum(spectrum) {
    let found_Idx = 0;
    for (let i = 0; i < ComplexCoherence.available_spectrum.length; i++) {
        if (ComplexCoherence.available_spectrum[i] === spectrum) {
            found_Idx = i;
            break;
        }
    }
    found_Idx = found_Idx % ComplexCoherence.AVAILABLE_COLORS.length;
    ComplexCoherence.hex_color = ComplexCoherence.AVAILABLE_COLORS[found_Idx].hex_color;
    ComplexCoherence.hex_face_color = ComplexCoherence.AVAILABLE_COLORS[found_Idx].hex_face_color;
}