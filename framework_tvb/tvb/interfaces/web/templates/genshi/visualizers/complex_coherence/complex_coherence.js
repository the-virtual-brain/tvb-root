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


var ComplexCoherence = {
    MARGIN: {top: 30, right: 20, bottom: 30, left: 50},
    AVAILABLE_COLORS: [{hex_color: '#0F94DB', hex_face_color: '#469EEB'},
        {hex_color: '#16C4B9', hex_face_color: '#0CF0E1'},
        {hex_color: '#CC4F1B', hex_face_color: '#FF9848'}],
    cohAvDataCurve: null,
    cohAreaDataCurve: null,
    svgContainer: null,
    xAxisScale: null,
    yAxisScale: null,
    xAxis: null,
    yAxis: null,
    xAxisLabel: null,
    yAxisLabel: null,
    plotTitle: null,
    xMin: null,
    xMax: null,
    yMin: null,
    yMax: null,
    available_spectrum: null,
    url_base: null,
    hex_color: null,
    hex_face_color: null
};

function Complex_complex_coherence_init(plotName, xAxisName, yAxisName, x_min, x_max, url_base, available_spectrum) {

    ComplexCoherence.xMin = x_min;
    ComplexCoherence.xMax = x_max;
    ComplexCoherence.url_base = url_base;
    ComplexCoherence.available_spectrum = $.parseJSON(available_spectrum);

    var svgContainer = d3.select("#svg-container");
    ComplexCoherence.svgContainer = svgContainer;

    var xAxisScale = d3.scale.linear();
    var yAxisScale = d3.scale.linear();

    ComplexCoherence.xAxisScale = xAxisScale;
    ComplexCoherence.yAxisScale = yAxisScale;

    ComplexCoherence.xAxis = d3.svg.axis()
        .scale(xAxisScale);
    ComplexCoherence.yAxis = d3.svg.axis()
        .scale(yAxisScale)
        .orient("left");

    var xAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .text(xAxisName);
    var plotTitle = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .text(plotName);
    var yAxisLabel = svgContainer.append("text")
        .attr("text-anchor", "middle")
        .text(yAxisName);

    ComplexCoherence.xAxisLabel = xAxisLabel;
    ComplexCoherence.yAxisLabel = yAxisLabel;
    ComplexCoherence.plotTitle = plotTitle;

}

function _Complex_loadData(coh_spec_sd, coh_spec_av) {
    var x_min = ComplexCoherence.xMin;
    var x_max = ComplexCoherence.xMax;
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

function Complex_drawGraph() {
    d3.selectAll("g").remove();
    d3.selectAll("path").remove();
    d3.selectAll("rect").remove();

    var svgContainer = ComplexCoherence.svgContainer;
    var xAxis = ComplexCoherence.xAxis;
    var yAxis = ComplexCoherence.yAxis;
    var xAxisScale = ComplexCoherence.xAxisScale;
    var yAxisScale = ComplexCoherence.yAxisScale;

    var width = svgContainer["0"]["0"].clientWidth - ComplexCoherence.MARGIN.left - ComplexCoherence.MARGIN.right,
        height = svgContainer["0"]["0"].clientHeight - ComplexCoherence.MARGIN.top - ComplexCoherence.MARGIN.bottom;

    xAxisScale.range([ComplexCoherence.MARGIN.left, width - ComplexCoherence.MARGIN.right]).domain([ComplexCoherence.xMin, ComplexCoherence.xMax]);
    yAxisScale.range([height - ComplexCoherence.MARGIN.top, ComplexCoherence.MARGIN.bottom]).domain([ComplexCoherence.yMin, ComplexCoherence.yMax]);
    xAxis.scale(xAxisScale);
    yAxis.scale(yAxisScale);

    svgContainer.append("svg:g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + (height - ComplexCoherence.MARGIN.bottom) + ")")
        .call(xAxis);

    svgContainer.append("svg:g")
        .attr("class", "y axis")
        .attr("transform", "translate(" + (ComplexCoherence.MARGIN.left) + ",0)")
        .call(yAxis);

    svgContainer.append('rect')
        .attr("height", height - ComplexCoherence.MARGIN.top - ComplexCoherence.MARGIN.bottom)
        .attr("width", width - ComplexCoherence.MARGIN.left - ComplexCoherence.MARGIN.right)
        .attr("transform", "translate(" + (ComplexCoherence.MARGIN.left) + "," + ComplexCoherence.MARGIN.bottom + ")")
        .attr("fill", "white");

    ComplexCoherence.xAxisLabel
        .attr("transform", "translate(" + width / 2 + "," + (height + 10) + ")");
    ComplexCoherence.yAxisLabel
        .attr("transform", "translate(10," + height / 2 + ")" + "rotate(-90)");
    ComplexCoherence.plotTitle
        .attr("transform", "translate(" + width / 2 + ",15)");

    _Complex_drawDataCurves();
}

function _Complex_drawDataCurves() {
    var cohAvDataCurve = ComplexCoherence.cohAvDataCurve;
    var cohAreaDataCurve = ComplexCoherence.cohAreaDataCurve;
    var svgContainer = ComplexCoherence.svgContainer;
    var xAxisScale = ComplexCoherence.xAxisScale;
    var yAxisScale = ComplexCoherence.yAxisScale;
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

    var lineGen = d3.svg.line()
        .x(function (d) {
            return xAxisScale(d[0]);
        })
        .y(function (d) {
            return yAxisScale(d[1]);
        })
        .interpolate("linear");


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

function Complex_changeXScale(xAxisScale) {
    var svgContainer = ComplexCoherence.svgContainer;
    var width = svgContainer["0"]["0"].clientWidth - ComplexCoherence.MARGIN.left - ComplexCoherence.MARGIN.right;
    var x_min = ComplexCoherence.xMin;
    var x_max = ComplexCoherence.xMax;
    if (xAxisScale === "log")
        ComplexCoherence.xAxisScale = d3.scale.log().range([ComplexCoherence.MARGIN.left, width - ComplexCoherence.MARGIN.right]).domain([x_min, x_max]);
    else
        ComplexCoherence.xAxisScale = d3.scale.linear().range([ComplexCoherence.MARGIN.left, width - ComplexCoherence.MARGIN.right]).domain([x_min, x_max]);
    Complex_drawGraph();
}

function Complex_getSpectrum(spectrum) {
    let url_base = ComplexCoherence.url_base;
    doAjaxCall({
        url: url_base + "selected_spectrum=" + spectrum,
        type: 'POST',
        async: true,
        success: function (data) {
            data = $.parseJSON(data);
            _Complex_loadData(data.coh_spec_sd, data.coh_spec_av);
            ComplexCoherence.yMin = data.ymin;
            ComplexCoherence.yMax = data.ymax;
            ComplexCoherence.yAxisScale.domain([data.ymin, data.ymax]);
            _Complex_updateColourForSpectrum(spectrum);
            Complex_drawGraph();
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