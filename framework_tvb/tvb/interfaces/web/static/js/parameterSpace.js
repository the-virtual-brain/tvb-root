/**
 * TheVirtualBrain-Framework Package. This package holds all Data Management, and 
 * Web-UI helpful to run brain-simulations. To use it, you also need do download
 * TheVirtualBrain-Scientific Package (for simulators). See content of the
 * documentation-folder for more details. See also http://www.thevirtualbrain.org
 *
 * (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
 *
 * This program is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU General Public License version 2 as published by the Free
 * Software Foundation. This program is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details. You should have received a copy of the GNU General 
 * Public License along with this program; if not, you can download it here
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0
 *
 **/
/* global doAjaxCall, displayMessage */

// We keep all-nodes information for current PSE as a global, to have them ready at node-selection, node-overlay.
var PSE_nodesInfo;
// Keep Plot-options and MIN/MAx colors for redraw (e.g. at resize).
var _PSE_plotOptions; 
var _PSE_minColor;
var _PSE_maxColor;
var _PSE_plot;

/*
 * @param canvasId: the id of the HTML DIV on which the drawing is done. This should have sizes defined or else FLOT can't do the drawing.
 * @param xLabels: the labels for the x - axis
 * @param yLabels: the labels for the y - axis
 * @param seriesArray: the actual data to be used by FLOT
 * @param data_info: additional information about each node. Used when hovering over a node
 * @param min_color: minimum color, used for gradient
 * @param max_color: maximum color, used for gradient
 * @param backPage: page where visualizers fired from overlay should take you back.
 */
function _updatePlotPSE(canvasId, xLabels, yLabels, seriesArray, data_info, min_color, max_color, backPage) {

    _PSE_minColor = min_color;
    _PSE_maxColor = max_color;
    PSE_nodesInfo = data_info;
    
    _PSE_plotOptions = {
        series: {
            lines: {
                show: false
            },
            points: {
                lineWidth: 0,
                show: true,
                fill: true
            }
        },
        xaxis: {
            min: -1,
            max: xLabels.length,
            tickSize: 1,
            tickFormatter: function(val) {
                if (val < 0 || val >= xLabels.length) {
                    return "";
                }
                return xLabels[val];
            }
        },
        yaxis: {
            min: -1,
            max: yLabels.length,
            tickSize: 1,
            tickFormatter: function(val) {
                if (val < 0 || val >= yLabels.length || yLabels[val] === "_") {
                    return "";
                }
                return yLabels[val];
            }
        },
        grid: {
            clickable: true,
            hoverable: true
        }

    };
    
    _PSE_plot = $.plot($("#" + canvasId), $.parseJSON(seriesArray), $.extend(true, {}, _PSE_plotOptions));
    changeColors();
    $(".tickLabel").each(function() { $(this).css("color", "#000000"); });

    //if you want to catch the right mouse click you have to change the flot sources
    // because it allows you to catch only "plotclick" and "plothover"
    applyClickEvent(canvasId, backPage);
    applyHoverEvent(canvasId);
}

/*
 * Do a redraw of the plot. Be sure to keep the resizable margin elements as the plot method seems to destroy them.
 */
function redrawPlot(plotCanvasId) {
    // todo: mh the selected element is not an ancestor of the second tab!!!
    // thus this redraw call fails, ex on resize
    if (_PSE_plot != null) {
        _PSE_plot = $.plot($('#'+plotCanvasId)[0], _PSE_plot.getData(), $.extend(true, {}, _PSE_plotOptions));
    }
}

/*
 * Fire DataType overlay when clicking on a node in PSE.
 */
function applyClickEvent(canvasId, backPage) {

    var currentCanvas = $("#"+canvasId);
    currentCanvas.unbind("plotclick");
    currentCanvas.bind("plotclick", function (event, pos, item) {
                if (item != null) {
                        var dataPoint = item.datapoint;
                        var dataInfo = PSE_nodesInfo[dataPoint[0]][dataPoint[1]];
                        if (dataInfo['dataType'] != undefined) {
                            displayNodeDetails(dataInfo['Gid'], dataInfo['dataType'], backPage);
                        }
                }
            });
}

var previousPoint = null;
/*
 * On hover display few additional information about this node.
 */
function applyHoverEvent(canvasId) {

    $("#" + canvasId).bind("plothover", function (event, pos, item) {
        if (item) {
            if (previousPoint != item.dataIndex) {
                previousPoint = item.dataIndex;
                $("#tooltip").remove();
                var dataPoint = item.datapoint;
                var dataInfo = PSE_nodesInfo[dataPoint[0]][dataPoint[1]];
                var tooltipText = ("" + dataInfo["tooltip"]).split("&amp;").join("&").split("&lt;").join("<").split("&gt;").join(">");

                $('<div id="tooltip"> </div>').html(tooltipText
                    ).css({ position: 'absolute', display: 'none', top: item.pageY + 5, left: item.pageX + 5,
                           border: '1px solid #fdd', padding: '2px', 'background-color': '#C0C0C0', opacity: 0.80 }
                    ).appendTo('body').fadeIn(200);
            }
        } else {
            $("#tooltip").remove();
            previousPoint = null;
        }
    });
}


function PSEDiscreteInitialize(labelsXJson, labelsYJson, series_array, dataJson, backPage, hasStartedOperations,
                               min_color, max_color, min_size, max_size) {

    //ColSch_initColorSchemeParams(min_color, max_color, changeColors);

    var labels_x = $.parseJSON(labelsXJson);
    var labels_y = $.parseJSON(labelsYJson);
    var data = $.parseJSON(dataJson);

    _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage);

    $('#minColorLabel')[0].innerHTML = '<mark>Minimum color metric</mark> ' + Math.round(min_color * 1000) / 1000;
    $('#maxColorLabel')[0].innerHTML = '<mark>Maximum color metris</mark> ' + Math.round(max_color * 1000) / 1000;
    $('#minShapeLabel')[0].innerHTML = '<mark>Minimum shape</mark> ' + Math.round(min_size * 1000) / 1000;
    $('#maxShapeLabel')[0].innerHTML = '<mark>Maximum shape</mark> ' + Math.round(max_size * 1000) / 1000;

    // Prepare functions for Export Canvas as Image
    var canvas = $("#main_div_pse").find(".flot-base")[0];
    canvas.drawForImageExport = function () {
                /* this canvas is drawn by FLOT library so resizing it directly has no influence;
                 * therefore, its parent needs resizing before redrawing;
                 * canvas.afterImageExport() is used to bring is back to original size */
                 var canvasDiv = $("#main_div_pse");
                 var oldHeight = canvasDiv.height();
                 var scale = C2I_EXPORT_HEIGHT / oldHeight;
                 canvas.oldStyle = canvasDiv.attr('style');

                 canvasDiv.css("display", "inline-block");
                 canvasDiv.width(canvasDiv.width() * scale);
                 canvasDiv.height(oldHeight * scale);
                 redrawPlot('main_div_pse');
    };
    canvas.afterImageExport = function() {
                // bring it back to original size and redraw
                var canvasDiv = $("#main_div_pse");
                canvasDiv.attr('style', canvas.oldStyle);
                redrawPlot('main_div_pse');
    };

    if (hasStartedOperations) {
        setTimeout("PSE_mainDraw('main_div_pse','" + backPage + "')", 3000);
    }
}


/*
 * Take currently selected metrics and refresh the plot. 
 */
function PSE_mainDraw(parametersCanvasId, backPage, groupGID) {

    if (groupGID == null) {
        // We didn't get parameter, so try to get group id from page
        groupGID = document.getElementById("datatype-group-gid").value;
    }
    if (backPage == null || backPage == '') {
        backPage = get_URL_param('back_page');
    }

    var url = '/burst/explore/draw_discrete_exploration/' + groupGID + '/' + backPage;
    var selectedColorMetric = $('#color_metric_select').val();
    var selectedSizeMetric = $('#size_metric_select').val();

    if (selectedColorMetric != '' && selectedColorMetric != null) {
        url += '/' + selectedColorMetric;
        if (selectedSizeMetric != ''  && selectedSizeMetric != null) {
            url += '/' + selectedSizeMetric;
        }
    }

    doAjaxCall({
            type: "POST",
            url: url,
            success: function(r) { 
                    $('#' + parametersCanvasId).html(r);
                },
            error: function() {
                displayMessage("Could not refresh with the new metrics.", "errorMessage");
            }});
}


/**
 * Changes the series colors according to the color picker component.
 */
function changeColors() {
    var series = _PSE_plot.getData();
    for (var i = 0; i < series.length; i++) {
        var indexes = series[i].datapoints.points;
        var dataInfo = PSE_nodesInfo[indexes[0]][indexes[1]];
        var colorWeight = dataInfo['color_weight'];
        var color = ColSch_getGradientColorString(colorWeight, _PSE_minColor, _PSE_maxColor);
        series[i].points.fillColor = color;
        series[i].color = color;
    }
    _PSE_plot.draw();
}


/*************************************************************************************************************************
 * 			ISOCLINE PSE BELLOW
 *************************************************************************************************************************/


var serverURL = null;
var figuresDict = null;
var currentFigure = null;


/*
 * Do the actual resize on currentFigure global var, and a given width and height.
 */
function resizePlot(width, height) {

    if (currentFigure != null) {
        MPLH5_resize = currentFigure;
        do_resize(currentFigure, width, height);
        MPLH5_resize = -1;
    }
}

/*
 * Store all needed data as js variables so we can use later on.
 */
function initISOData(metric, figDict, servURL) {

    figuresDict = $.parseJSON(figDict);
    serverURL = servURL;
    currentFigure = figuresDict[metric];
    connect_manager(serverURL, figuresDict[metric]);
    $('#' + metric).show();
    initMPLH5CanvasForExportAsImage(figuresDict[metric]);
}

/*
 * On plot change update metric and do any required changes like resize on new selected plot.
 */
function updateMetric(selectComponent) {

    var newMetric = $(selectComponent).find(':selected').val();
    showMetric(newMetric);
    var pseElem = $('#section-pse');
    var width = pseElem.width() - 60;
    var height = pseElem.height() - 90;
    waitOnConnection(currentFigure, 'resizePlot('+ width +', '+ height +')', 200, 50);
}

/*
 * Update html to show the new metric. Also connect to backend mplh5 for this new image.
 */
function showMetric(newMetric) {

    for (var key in figuresDict) {
        $('#' + key).hide()
            .find('canvas').each(function () {
                if (this.drawForImageExport) {            // remove redrawing method such that only current view is exported
                    this.drawForImageExport = null;
                }
            });
    }
    currentFigure = figuresDict[newMetric];
    connect_manager(serverURL, figuresDict[newMetric]);
    $('#' + newMetric).show();
    initMPLH5CanvasForExportAsImage(figuresDict[newMetric]);
}

/*
 * This is the callback that will get evaluated by an onClick event on the canvas through the mplh5 backend.
 */
function clickedDatatype(datatypeGid) {

    displayNodeDetails(datatypeGid);
}

/*
 * Update info on mouse over. This event is passed as a callback from the isocline python adapter.
 */
function hoverPlot(id, x, y, val) {

    document.getElementById('cursor_info_' + id).innerHTML = 'x axis:' + x + ' y axis:' + y + ' value:' + val;
}


function Isocline_MainDraw(groupGID, divId, width, height) {
    width = Math.floor(width);
    height = Math.floor(height);
    $('#' + divId).html('');
    doAjaxCall({
            type: "POST",
            url: '/burst/explore/draw_isocline_exploration/' + groupGID + '/' + width + '/' + height,
            success: function(r) {
                    $('#' + divId).html(r);
                },
            error: function() {
                displayMessage("Could not refresh with the new metrics.", "errorMessage");
    }});
}


