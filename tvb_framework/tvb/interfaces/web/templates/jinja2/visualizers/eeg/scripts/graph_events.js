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
* Handle zooming speed and scale related settings for animated graph.
**/
//The zoom stack used for keeping track of zoom events for the 'back' option
var zoomStack = [];
//Previously point when displaying info on mouse hover
var previousPoint = null;

function initializeCanvasEvents() {
    // Prepare functions for Export Canvas as Image
    var canvas = $("#EEGcanvasDiv .flot-base")[0];

    canvas.drawForImageExport = function () {
        /* this canvas is drawn by FLOT library so resizing it directly has no influence;
         * therefore, its parent needs resizing before redrawing;
         * canvas.afterImageExport() is used to bring is back to original size */
         var canvasDiv = $("#EEGcanvasDiv");
         var oldHeight = canvasDiv.height();
         canvas.scale = C2I_EXPORT_HEIGHT / oldHeight;

         canvasDiv.width(canvasDiv.width() * canvas.scale);
         canvasDiv.height(oldHeight * canvas.scale);

         redrawPlot(plot.getData());
    };
    canvas.afterImageExport = function() {
        // bring it back to original size and redraw
        var canvasDiv = $("#EEGcanvasDiv");
        canvasDiv.width(canvasDiv.width() / canvas.scale);
        canvasDiv.height(canvasDiv.height() / canvas.scale);
        redrawPlot(plot.getData());
    };

    $("#EEGcanvasDiv").resizable({
         alsoResize: canvas,
         stop: function(e, ui) {
             redrawPlot(plot.getData());
         }
    });
}


//------------------------------------------------START ZOOM RELATED CODE--------------------------------------------------------
function bindZoomEvent() {
    /*
     * On a zoom event, retain the x and y axis values in a stack, for the 'Back' zoom possibility.
     */
    $("#EEGcanvasDiv").bind('plotzoom', function (event, plot) {
        var axes = plot.getAxes();
        AG_isSpeedZero = true;
    });
    
    $("#EEGcanvasDiv").bind('plotselected', function (event, ranges) {
        zoomStack.push([AG_options['xaxis']['min'], AG_options['xaxis']['max'], AG_options['yaxis']['min'], AG_options['yaxis']['max']]);
        AG_options['xaxis'] = { min: ranges.xaxis.from, max: ranges.xaxis.to };
        AG_defaultYaxis['min'] = ranges.yaxis.from;
        AG_defaultYaxis['max'] = ranges.yaxis.to;
        //AG_options['yaxis'] = { min: ranges.yaxis.from, max: ranges.yaxis.to }
        AG_isSpeedZero = true;
        redrawPlot(plot.getData());
    });
}

function stopAnimation() {
    AG_isStopped = !AG_isStopped;
    var btn = $("#ctrl-action-pause");
    if (AG_isStopped) {
        btn.html("Start");
        btn.attr("class", "action action-controller-launch");
    } else {
        btn.html("Pause");
        btn.attr("class", "action action-controller-pause");
    }
    if (!AG_isStopped) {
        drawGraph(true, noOfShiftedPoints);
    }
}

function resetToDefaultView() {
    /*
     * When resetting to default view, clear all the data from the zoom stack
     * and set the home values for x and y values.
     */
    AG_options.xaxis = AG_homeViewXValues;
    zoomStack = [];
    AG_defaultYaxis.min = AG_homeViewYValues[0];
    AG_defaultYaxis.max = AG_homeViewYValues[1];
    redrawPlot(plot.getData());
    if (!isSmallPreview ) {
        if ($("#ctrl-input-speed").slider("option", "value") != 0) {
            AG_isSpeedZero = false;
        }
   }
}


function zoomBack() {
    /*
     * Pop the last entry from the zoom stack and redraw with those option.
     */
    if (zoomStack.length > 1) {
        var previousValues = zoomStack.pop();
        AG_options['xaxis'] = {min: previousValues[0], max: previousValues[1]};
        AG_defaultYaxis['min'] = previousValues[2];
        AG_defaultYaxis['max'] = previousValues[3];
        redrawPlot(plot.getData());
    } else {
        resetToDefaultView()
    }
}

//------------------------------------------------END ZOOM RELATED CODE--------------------------------------------------------

//------------------------------------------------START SCALE RELATED CODE--------------------------------------------------------
/**
 * If we change the AG_translationStep then we have to redraw the current view using the new value of the AG_translationStep
 */
function redrawCurrentView() {
    var diff = AG_currentIndex - AG_numberOfVisiblePoints;
    for (var k = 0; k < AG_numberOfVisiblePoints; k++) {
        AG_displayedTimes[k] = AG_time[k + diff];
        for (var i = 0; i < AG_noOfLines; i++) {
            AG_displayedPoints[i][k] = [AG_time[k + diff], AG_addTranslationStep(AG_allPoints[i][k + diff], i)];
        }
    }
    AG_createYAxisDictionary(AG_noOfLines);
    redrawPlot([]);
}


function drawSliderForScale() {
    function _onchange(){
        /** When scaling, we need to redraw the graph and update the HTML with the new values.
         */
        var spacing = $("#ctrl-input-spacing").slider("value") / 4;
        var scale = $("#ctrl-input-scale").slider("value");

        if (spacing >= 0 && AG_currentIndex <= AG_numberOfVisiblePoints) {
            AG_currentIndex = AG_numberOfVisiblePoints;
        } else if (spacing < 0 && (AG_allPoints[0].length - AG_currentIndex) < AG_numberOfVisiblePoints) {
            AG_currentIndex = AG_allPoints[0].length;
        }
        AG_displayedPoints = [];
        for (var i = 0; i < AG_noOfLines; i++) {
            AG_displayedPoints.push([]);
        }
        resetToDefaultView();
        _updateScaleFactor(spacing, scale);
    }

    $("#ctrl-input-spacing").slider({ value: 4, min: 0, max: 8, change: _onchange});
    $("#ctrl-input-scale").slider({ value: 1, min: 1, max: 32, change: _onchange});

    $("#display-spacing").html("" + AG_translationStep + '*' +AG_computedStep.toFixed(2));
    $("#display-scale").html("" + AG_scaling);
}


function _updateScaleFactor(spacing, scale) {
    AG_translationStep = spacing;
    AG_scaling = scale;
    $("#display-spacing").html("" + AG_translationStep + '*' +AG_computedStep.toFixed(2));
    $("#display-scale").html("" + AG_scaling);
    redrawCurrentView();
    if (AG_isStopped) {
        refreshChart();
    }
}

//------------------------------------------------END SCALE RELATED CODE--------------------------------------------------------

//------------------------------------------------START HOVER RELATED CODE--------------------------------------------------------

function bindHoverEvent() {
        $("#EEGcanvasDiv").bind("plothover", function (event, pos, item) {
        /*
         * When hovering over plot, if an item (FLOT point) is hovered over, then find the channel of that point
         * by means of using the number of AG_translationStep * AG_computedStep intervals from the first channel.
         * Then using this and the apporximate of the time value, get the actual data value from the AG_allPoints array.
         */
        if (item) {
            var timeValue = pos.x.toFixed(4);
            var dataValue = pos.y.toFixed(4);
            var rowIndex = AG_channelColorsDict[item.series.color];
            if (rowIndex == undefined) {
                $("#info-channel").html(' None');
                $("#info-time").html(" 0");
                $("#info-value").html(" 0");
                $("#tooltip").remove();
                previousPoint = null;
                return;
            }
            var startTime = AG_time.indexOf(AG_displayedPoints[0][0][0]);
            dataValue = AG_allPoints[rowIndex][startTime + (parseInt((timeValue - AG_displayedPoints[0][0][0]) / (AG_displayedPoints[0][1][0] - AG_displayedPoints[0][0][0])))];
            $("#info-channel").html(' ' + chanDisplayLabels[displayedChannels[rowIndex]]);
            $("#info-time").html(" " + timeValue);
            $("#info-value").html(" " + dataValue);

            if (previousPoint != item.dataIndex || dataValue != undefined) {
                previousPoint = item.dataIndex;

                $("#tooltip").remove();
                var x = item.datapoint[0].toFixed(2),
                    y = item.datapoint[1].toFixed(2);
                showTooltip(item.pageX, item.pageY,
                            "Time: " + timeValue + ", Value: " + dataValue);
            }
        } else {
            $("#info-channel").html(' None');
            $("#info-time").html(" 0");
            $("#info-value").html(" 0");
            $("#tooltip").remove();
            previousPoint = null;
        }
    });
}

function showTooltip(x, y, contents) {
    /*
     * A tooltip to display information about a specific point on 'mouse over'.
     */
    $('<div id="tooltip">' + contents + '</div>').css( {
        position: 'absolute',
        display: 'none',
        top: y + 5,
        left: x + 5,
        border: '1px solid #fdd',
        padding: '2px',
        'background-color': '#fee',
        opacity: 0.80
    }).appendTo("body").fadeIn(200);
}
//------------------------------------------------END HOVER RELATED CODE--------------------------------------------------------

//------------------------------------------------START SPEED RELATED CODE--------------------------------------------------------
/**
 * The method should be used when the animated chart is stopped. Draw graph without shifting.
 */
function refreshChart() {
    AG_isStopped = false;
    drawGraph(false, noOfShiftedPoints);
    AG_isStopped = true;
}

function drawSliderForAnimationSpeed() {
    $("#ctrl-input-speed").slider({
        orientation: 'horizontal',
        value: 3,
        min: -50,
        max: 50,
        change: function(event, ui) {
            resetToDefaultView();
            updateSpeedFactor();
        }
    });
}


function updateSpeedFactor() {
    var speed = $("#ctrl-input-speed").slider("option", "value");
    $('#display-speed').html(''+ speed);
    AG_isSpeedZero = (speed == 0);
}

//------------------------------------------------END SPEED RELATED CODE--------------------------------------------------------
