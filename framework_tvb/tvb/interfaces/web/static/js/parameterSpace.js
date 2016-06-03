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
        margins: { // is this the correct way to be doing margins? It's just how I have in the past,
            top: 20,
            bottom: 60,
            left: 70,
            right: 20
        },
        xaxis: {
            labels: xLabels, // is there a better way to get access to these values inside my plotting?
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
            labels: yLabels,
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
    var _d3PSE_plot = d3Plot("#" + canvasId, $.parseJSON(seriesArray), $.extend(true, {}, _PSE_plotOptions));

    //this has been commented out below so that I can see what I have done on the canvas after the above function has ended
    /*_PSE_plot = $.plot($("#" + canvasId), $.parseJSON(seriesArray), $.extend(true, {}, _PSE_plotOptions));
    changeColors(); // this will need to eventually have the addition of the d3 plot function
    $(".tickLabel").each(function () {
        $(this).css("color", "#000000");
    });
    //if you want to catch the right mouse click you have to change the flot sources
    // because it allows you to catch only "plotclick" and "plothover"
    applyClickEvent(canvasId, backPage);
     applyHoverEvent(canvasId);*/
}


function d3Plot(placeHolder, data, options) {
//todo check to see whether there is already a canvas of the d3 variety because then we can just use that if redraw must happen

    function createScale(xORy) {
        // should I incorporate some sort of testing for values before actually getting into the function?
        if (xORy === "x") {
            var newScale = d3.scale.linear()
                .domain(d3.extent(options.xaxis.labels))
                .range([options.margins.left, canvasDimensions.w - options.margins.right]);
            return newScale
        } else {
            newScale = d3.scale.linear()
                .domain(d3.extent(options.yaxis.labels))
                .range([canvasDimensions.h - (options.margins.bottom), options.margins.top]);
            return newScale
        }
    }

    function createAxis(xORy) {
        if (xORy === "x") { // should I be creating the whole axis inside here, or should I simply return the axis that has the parts to be customized and called later
            newAxis = d3.svg.axis().scale(xScale)
                .orient("bottom")
                .ticks(options.xaxis.max);
            return newAxis
        }
        else {
            newAxis = d3.svg.axis().scale(yScale)
                .orient("left")
                .ticks(options.yaxis.max);
            return newAxis
        }
    }

    function dataToOpt(checkd, xORy) {
        if (xORy === "x") {
            return options.xaxis.tickFormatter(checkd.data[0][0]);
        } else
            return options.yaxis.tickFormatter(checkd.data[0][1])

    }

    function brushed() {
        var extent = brush.extent();
        circles.classed("selected", function (d) {
            return extent[0][0] <= d.data[0][0] && d.data[0][0] <= extent[1][0] // basically says that circle x score, is inbetween brush x bounds and vice verse for y
                &&
                extent[0][1] <= d.data[0][1] && d.data[0][1] <= extent[1][1]
        })

    }

    function brushend() {
        var extent = brush.extent();
        xScale.domain(brush.empty() ? xRef.domain() : [extent[0][0], extent[1][0]]);
        yScale.domain(brush.empty() ? yRef.domain() : [extent[0][1], extent[1][1]]);

        moveDots();
        replaceAxes();

        d3.select(".brush").call(brush.clear())
    }

    function moveDots() {
        circles
            .transition()
            .delay(500)
            .attr({
                cx: function (d) {
                    return xScale(dataToOpt(d, "x"))
                },
                cy: function (d) {
                    return yScale(dataToOpt(d, "y"));
                    // return yScale(d.yCen) // why is this placing dots far below the bottom of the pane? Is the canvas dimension off?
                }

            })
    }

    function replaceAxes() {
        canvas.transition().duration(500)
            .select("#xAxis")
            .call(xAxis);

        canvas.transition().duration(500)
            .select("#yAxis")
            .call(yAxis);

    }

    var myBase, canvasDimensions, canvas, xScale, yScale, xRef, yRef, xAxis, yAxis, circles, brush;
    myBase = d3.select(placeHolder);
    canvasDimensions = {h: parseInt(myBase.style("height")), w: parseInt(myBase.style("width"))};
    canvas = myBase.append("svg") //todo must make plottable canvas be inbetween axes, otherwise zoom adjusted circles can be seen outside of rational graphing area
        .attr({
            height: canvasDimensions.h,
            width: canvasDimensions.w
        });
    xScale = createScale("x");
    yScale = createScale("y");
    xRef = xScale.copy();
    yRef = yScale.copy();
    xAxis = createAxis("x");
    yAxis = createAxis("y");
    circles = canvas.selectAll("circle").data(data).enter().append("circle")
        .attr({
            r: function (d) {
                return d.points.radius * 1.25
            },
            cx: function (d) {
                return xScale(dataToOpt(d, "x"))
            },
            cy: function (d) {
                return yScale(dataToOpt(d, "y"));
                // return yScale(d.yCen) // why is this placing dots far below the bottom of the pane? Is the canvas dimension off?
            }

        });
    brush = d3.svg.brush()
        .x(xScale)
        .y(yScale)
        .on("brush", brushed)
        .on("brushend", brushend);


    canvas.append("g")
        .attr("id", "xAxis")
        .attr("transform", "translate (0," + (canvasDimensions.h - 35) + ")")
        .call(xAxis);
    canvas.append("g")
        .attr("id", "yAxis")
        .attr("transform", "translate (" + (options.margins.left - 20) + ",0)")
        .call(yAxis);
    // this is now the area that should allow for drawing the lines of the grid
    canvas.append("g") //todo ask about the css that needs to be added here so that the grid lines show up.
        .attr("id", "x-grid")
        .attr("transform", "translate(15," + (canvasDimensions.h - 35) + ")") //figure out a way to make sure that the lines land on the dots
        .style("stroke", "black")
        .call(createAxis("x")
            .tickSize(-canvasDimensions.h, 0, 0)
            .tickFormat(""));
    canvas.append("g") //todo ask about the css that needs to be added here so that the grid lines show up.
        .attr("id", "y-grid")
        .attr("transform", "translate (" + (options.margins.left - 20) + ",15)") //figure out a way to make sure that the lines land on the dots
        .style("stroke", "black")
        .call(createAxis("y")
            .tickSize(-canvasDimensions.w, 0, 0)
            .tickFormat(""));
    //todo again visual grid stuff. How should I go about making the grid fit the canvas better?


    d3.select("#Magnify").on("click", function (d) {
        var activeBrush = d3.select(".brush")
        if (activeBrush.empty() == true) {
            canvas.append("g")
                .attr("class", "brush")
                .call(brush)
                .selectAll("rect");
        } else {
            activeBrush.remove()
        }
    //.attr("height", canvasDimensions.h - (options.margins.top + options.margins.bottom))

    })


    d3.selectAll("circle").on("mouseover", function (d) { // why can't a access the options variable inside this scope?
        var xVal, yVal;
        debugger;

    });

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

    var labels_x = $.parseJSON(labelsXJson);
    var labels_y = $.parseJSON(labelsYJson);
    var data = $.parseJSON(dataJson);

    min_color = parseFloat(min_color);
    max_color = parseFloat(max_color);
    min_size = parseFloat(min_size);
    max_size = parseFloat(max_size);

    ColSch_initColorSchemeGUI(min_color, max_color, function(){
        _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage);
    });

    function _fmt_lbl(sel, v){
        $(sel).html( Number.isNaN(v) ? 'not available': toSignificantDigits(v, 3));
    }

    _fmt_lbl('#minColorLabel', min_color);
    _fmt_lbl('#maxColorLabel', max_color);
    _fmt_lbl('#minShapeLabel', min_size);
    _fmt_lbl('#maxShapeLabel', max_size);

    if (Number.isNaN(min_color)){
        min_color = 0; max_color = 1;
    }
    _updatePlotPSE('main_div_pse', labels_x, labels_y, series_array, data, min_color, max_color, backPage);

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
        }
    });
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
    $('#cursor_info_' + id).html('x axis:' + x + ' y axis:' + y + ' value:' + val);
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
        }
    });
}


