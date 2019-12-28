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

// todo: do not use dom (hidden inputs) for data storage

var plot = null;

function drawHistogram(canvasDivId, data, labels, colorsPy) {
    var colors = computeColors(colorsPy);
    var histogramLabels = [];
    var histogramData = [];
    for (var i = 0; i < data.length; i++) {
        histogramData.push({data: [[i, parseFloat(data[i])]], color: colors[i]});
        histogramLabels.push([i, labels[i]]);
    }

    var options = {
        series: {stack: 0,
                 lines: {show: false, steps: false },
                 bars: {show: true, barWidth: 0.9, align: 'center', fill:0.8, lineWidth:0}},
        xaxis: {ticks: histogramLabels,
                labelWidth: 100}
    };

    var canvasDiv = $("#" + canvasDivId);
    plot = $.plot(canvasDiv, histogramData, options);


    // Prepare functions for Export Canvas as Image
    var canvas = $("#histogramCanvasId").find("canvas.flot-base")[0];
    canvas.drawForImageExport = function () {
                /* this canvas is drawn by FLOT library so resizing it directly has no influence;
                 * therefore, its parent needs resizing before redrawing;
                 * canvas.afterImageExport() is used to bring is back to original size */
                 var oldHeight = canvasDiv.height();
                 canvas.scale = C2I_EXPORT_HEIGHT / oldHeight;
                 canvasDiv.width(canvasDiv.width() * canvas.scale);
                 canvasDiv.height(oldHeight * canvas.scale);

                 plot = $.plot(canvasDiv, plot.getData(), options);
    };
    canvas.afterImageExport = function() {
                // bring it back to original size and redraw
                canvasDiv.width(canvasDiv.width() / canvas.scale).width("95%");         // set it back to percentage so
                canvasDiv.height(canvasDiv.height() / canvas.scale).height("90%");      // it updates on window resize

                plot = $.plot(canvasDiv, plot.getData(), options);
    };
}

function _drawHistogramLegend() {
    var legendDiv = $("#histogramLegend");
    ColSch_updateLegendColors(legendDiv[0], legendDiv.height() - 20);                    // -20 because of style

    // draw the labels
    var minValue = parseFloat($('#colorMinId').val()), maxValue = parseFloat($('#colorMaxId').val());
    ColSch_updateLegendLabels($(legendDiv), minValue, maxValue, legendDiv.height() - 20)
}

function computeColors(colorsArray) {
    // Compute actual colors from input array of numbers.
    var minColor = parseFloat($('#colorMinId').val());
    var maxColor = parseFloat($('#colorMaxId').val());
    var result = [];
    for (var i = 0; i < colorsArray.length; i++) {
        var color = ColSch_getGradientColorString(colorsArray[i], minColor, maxColor);
        result.push(color);
    }
    return result;
}

function changeColors() {
    var originalColors = $('#originalColors').val().replace('[','').replace(']','').split(',');
    var newColors = computeColors(originalColors);
    var data = plot.getData();
    for (var i = 0; i < data.length; i++) {
        data[i].color = newColors[i];
    }
    plot.draw();
    _drawHistogramLegend();
}

function startHistogramView(minColor, maxColor, data, labels, colors){
    function _draw(){
        drawHistogram('histogramCanvasId', data, labels, colors, '${xposition}');
        _drawHistogramLegend();
    }

    ColSch_initColorSchemeGUI(minColor, maxColor, changeColors);
    _draw();

    $(window).resize(function() {
        clearTimeout(this.resizingTimeout);
        this.resizingTimeout = setTimeout(function() {      // set timeout so it's only resized on finish
            _draw();
        }, 250);
    });
}