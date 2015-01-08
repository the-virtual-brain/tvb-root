var nodeColorRGB = [255, 255, 255];

var _colorScheme = null;                                             // the color scheme to be used for current drawing
var _linearGradientStart = 0, _linearGradientEnd = 1 ;               // keep the interest interval
var _sparseColorNo = 50;
var _refreshCallback = null ;                                        // this is called when color scheme changes update the visualiser
var SPARSE_COLORS_LENGTH = 80;

function drawSimpleColorPicker(divId, refreshFunction) {
    $('#' + divId).ColorPicker({
        color: '#ffffff',
        onShow: function (colpkr) {
            $(colpkr).fadeIn(500);
            return false;
        },
        onHide: function (colpkr) {
            $(colpkr).fadeOut(500);
            return false;
        },
        onChange: function (hsb, hex, rgb) {
            $('#' + divId + ' div').css('backgroundColor', '#' + hex);
            nodeColorRGB = [parseInt(rgb.r), parseInt(rgb.g), parseInt(rgb.b)];
            if (refreshFunction) {
                 refreshFunction();
            }
        }
    });	
    $('#' + divId + ' div').css('backgroundColor', '#ffffff');
}


function getNewNodeColor() {
	return nodeColorRGB;
}

function ColSch_getGradientColorString(pointValue, min, max) {
    var rgb_values = getGradientColor(pointValue, min, max);
    return "rgb(" + Math.round(rgb_values[0]*255) + "," + Math.round(rgb_values[1]*255) + "," + Math.round(rgb_values[2]*255) + ")";
}

function clampValue(value) {
    if (value > 1) {
        return 1;
    } else if (value < 0) {
        return 0;
    }

    return value;
}

// ================================= COLOR SCHEME FUNCTIONS START =================================

/**
 * Sets the current color scheme to the given one
 * @param scheme The color scheme to use; currently supported: 'linear', 'rainbow', 'hotcold', 'TVB', 'sparse'
 *               'light-hotcold', 'light-TVB'
 * @param notify_server When TRUE, trigger an ajax call to store on the server changed setting.
 */
function ColSch_setColorScheme(scheme, notify_server) {
    $('fieldset[id^="colorSchemeFieldSet_"]').hide();
    if(notify_server){
        //could throttle this
        doAjaxCall({
            url: '/user/set_viewer_color_scheme/' + scheme,
            error: function(jqXHR, textStatus, error){
                console.warn(error);
            }
        });
    } else {
        $("#setColorSchemeSelId").val(scheme);
    }
    _colorScheme = scheme;
    $("#colorSchemeFieldSet_" + scheme).show();
    if (_refreshCallback) {
        _refreshCallback();
    }
}

function ColSch_loadInitialColorScheme(async){

    doAjaxCall({
        url:'/user/get_viewer_color_scheme',
        async: async,
        success:function(data){
            ColSch_setColorScheme(JSON.parse(data), false);
        },
        error: function(jqXHR, textStatus, error){
            console.warn(error);
            ColSch_setColorScheme(null, false);
        }
    });
}

/**
 * Initialises the settings for all color schemes
 *
 * @param minValue The minimum value for the linear slider
 * @param maxValue The maximum value for the linear slider
 * @param [refreshFunction] A reference to the function which updates the visualiser
 */
function ColSch_initColorSchemeParams(minValue, maxValue, refreshFunction) {
    _refreshCallback = refreshFunction;
    // initialise the linear params
    var elemSliderSelector = $("#rangerForLinearColSch");
    if (elemSliderSelector.length < 1){
        displayMessage("Color scheme component not found for initialization...", "warningMessage");
        return;
    }
    elemSliderSelector.slider({
        range: true, min: minValue, max: maxValue, step: 0.001,
        values: [minValue, maxValue],
        slide: function(event, ui) {                            // update the UI
                $("#sliderMinValue").html(ui.values[0].toFixed(3));
                $("#sliderMaxValue").html(ui.values[1].toFixed(3));
        },
        change: function(event, ui) {
            _linearGradientStart = (ui.values[0] - minValue) / (maxValue - minValue);    // keep the interest interval
            _linearGradientEnd   = (ui.values[1] - minValue) / (maxValue - minValue) ;   // as normalized dist from min
            if (_refreshCallback) { _refreshCallback() }
        }
    });
    $("#sliderMinValue").html(minValue.toFixed(3));
    $("#sliderMaxValue").html(maxValue.toFixed(3));
    _linearGradientStart = 0;            // on start the whole interval is selected
    _linearGradientEnd   = 1;

    // initialise the sparse params
    var colorNoUIElem = $("#ColSch_colorNo");                    // cache the jQuery selector
    colorNoUIElem.html(_sparseColorNo);
    $("#sliderForSparseColSch").slider({
        min: 2, max: SPARSE_COLORS_LENGTH, step: 1, values: [_sparseColorNo],
        slide: function (event, ui) { colorNoUIElem.html(ui.value); },
        change: function (event, ui) {
            _sparseColorNo = ui.value;
            if (_refreshCallback) { _refreshCallback(); }
        }
    });
    if (!_colorScheme) {                      // on very first call, set the default color scheme
        ColSch_loadInitialColorScheme(true);
    }
}

/**
 * Factory function which returns a color for the given point in interval (min, max),
 * according to the current <code>_colorScheme</code>
 *
 * @param pointValue The value whose corresponding color is returned
 * @param max   The maximum value of the array, to use for computing gradient
 * @param min   Maximum value in colors array.
 *
 * NOTE: The following condition should be true: <code> min <= pointValue <= max </code>
 */
function getGradientColor(pointValue, min, max) {
    if (min == max)         // the interval is empty, so start color is the only possible one
        return getRainbowColor(0);
    if (pointValue < min)
        pointValue = min;   // avoid rounding problems
    if (pointValue > max)
        pointValue = max;

    var normalizedValue = (pointValue - min) / (max - min);

    return getRainbowColor(normalizedValue);
}

/**
 * 3d viewer styling
 * Css class like idea. In the future we might move these to css
 * class -> prop value list
 */
var ColSchDarkTheme = {
    connectivityStepPlot : {
        lineColor: [0.1, 0.1, 0.2],
        noValueColor: [0.0, 0.0, 0.0],
        backgroundColor: [0.05, 0.05, 0.05, 1.0],
        outlineColor: [0.3, 0.3, 0.3],
        selectedOutlineColor: [0.2, 0.2, 0.8]
    },
    connectivityPlot : {
        backgroundColor: [0.05, 0.05, 0.05, 1.0]
    },
    surfaceViewer : {
        backgroundColor: [0.05, 0.05, 0.05, 1.0],
        mutedRegionColor : [0.1, 0.1, 0.1]
            //, boundaryLineColor
    //, navigatorColor
    //, faceColor
    //, ambientLight
    }
};

var ColSchLightTheme = {
    connectivityStepPlot : {
        lineColor: [0.7, 0.7, 0.8],
        noValueColor: [0.9, 0.9, 0.9],
        backgroundColor: [1.0, 1.0, 1.0, 1.0],
        outlineColor: [0.5, 0.5, 0.5],
        selectedOutlineColor: [0.4, 0.4, 0.7]
    },
    connectivityPlot : {
        backgroundColor: [1.0, 1.0, 1.0, 1.0]
    },
    surfaceViewer : {
        backgroundColor: [1.0, 1.0, 1.0, 1.0],
        mutedRegionColor : [0.8, 0.8, 0.8]
    }
};

var ColSchTransparentTheme = {
    connectivityStepPlot : {
        lineColor: [0.7, 0.7, 0.8],
        noValueColor: [0.9, 0.9, 0.9],
        backgroundColor: [0.0, 0.0, 0.0, 0.0],
        outlineColor: [0.5, 0.5, 0.5],
        selectedOutlineColor: [0.4, 0.4, 0.7]
    },
    connectivityPlot : {
        backgroundColor: [0.0, 0.0, 0.0, 0.0]
    },
    surfaceViewer : {
        backgroundColor: [0.0, 0.0, 0.0, 0.0],
        mutedRegionColor : [0.8, 0.8, 0.8]
    }
};

/**
 * For each color scheme return a 3d theme
 */
function ColSchGetTheme(){
    return {
        linear : ColSchDarkTheme,
        TVB : ColSchDarkTheme,
        rainbow : ColSchDarkTheme,
        hotcold : ColSchDarkTheme,
        sparse: ColSchDarkTheme,
        lightHotcold : ColSchLightTheme,
        lightTVB : ColSchLightTheme,
        transparentHotCold : ColSchTransparentTheme
    }[_colorScheme||'linear'];
}

//texture coordinates of a color band
//todo these are very temporary. integrate into themes
var colorSchemeId = 0.3;
var mutedColorSchemeId = 0.5;
var measurePointsColorSchemeId = 0.2;

/**
 * Factory function which computes the colors for a whole array of values in interval (min, max)
 * according to the current <code>_colorScheme</code>
 *
 * @param {Array} values The values for which the colors are generated;
 *                       Condition: min <= values[i] <= max (for every values[i])
 * @param {Float} max   The maximum value of the array, to use for computing gradient
 * @param {Float} min   Maximum value in colors array.
 * @param {Float32Array} outputArray If specified, this is filled with the computed
 *                     Condition: outputArray.length = 4 * values.length (RGBA colors)
 * @returns {Array} If <code>outputArray</code> was not specified, a normal array is returned;
 *                  The empty array is returned otherwise
 */
function getGradientColorArray(values, min, max, outputArray) {
    var result = [], color = [];
    for (var i = 0; i < values.length; ++i) {
        color = getGradientColor(values[i], min, max);
        color.push(1);                               // add the alpha value
        if (outputArray)
            outputArray.set(color, i * 4);
        else
            result.concat(color);
    }

    return result;
}

/**
 * Returns an [r, g, b] color in the rainbow color scheme
 */
function getRainbowColor(normalizedValue) {
    normalizedValue *= 4;
    var r = Math.min(normalizedValue - 1.5, - normalizedValue + 4.5);
    var g = Math.min(normalizedValue - 0.5, - normalizedValue + 3.5);
    var b = Math.min(normalizedValue + 0.5, - normalizedValue + 2.5);

    return [clampValue(r), clampValue(g), clampValue(b)];
}

/**
 * Used by discrete color schemes. This makes the color interval open at the right end.
 * The effect is that the maximal value will map to the one below it. Otherwise we have
 * the corner case that a color is used by a single value leading to special handling.
 * @returns: a number within [0,1[
 */
function __convert_to_open(normalizedValue){
    var epsilon = 0.00001; // This assumes that the sparse color scheme has less than 10**6 colors.
    if (normalizedValue == 1.0) {
        normalizedValue = 1.0 - epsilon;
    }
    return normalizedValue;
}

// ================================= COLOR SCHEME FUNCTIONS  END  =================================

// ================================= LEGEND UPDATING FUNCTION START  =================================
/**
 * Function that should draw a gradient used for a legend. If the canvas for drawing doesn't exist, it will be created
 * @param containerDiv The div where the canvas is drawn
 * @param height The height of the drawn canvas
 */
function ColSch_updateLegendColors(containerDiv, height) {
    // Get canvas, or create it if it does not  exist
    var canvas = $(containerDiv).children("canvas");
    if (!canvas.length) {
        canvas = $('<canvas width="20">');
        $(containerDiv).append(canvas);
    }
    canvas.attr("height", height);
    canvas = canvas[0];

    var ctx = canvas.getContext('2d');
    var legendGranularity = 127;
    var step = height / legendGranularity;
    // y axis is inverted, so start from top
    var lingrad = ctx.createLinearGradient(0, height, 0, 0);

    for (var i = 0; i <= height ; i += step){
        lingrad.addColorStop(i / height, ColSch_getGradientColorString(i, 0, height));
    }
    ctx.fillStyle = lingrad;                            // Fill a rect using the gradient
    ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
}

function ColSch_updateLegendLabels(container, minValue, maxValue, height) {
    var table = $(container).is("table") ? $(container) : $(container).find("table");    // get the table
    if (!table.length)
        table = $("<table>").appendTo(container);                            // create one if it doesn't exist
    table.height(height);                                                    // set its height
    var legendLabels = $(table).find("td");                                  // search for td
    if (!legendLabels.length)                                               // if none is found, assume tr also don't exist
        legendLabels = $(table).append("<tr><tr><tr><tr><tr><tr>")          // so add 6 rows
            .find("tr").each(function(idx, elem) {                          // get them and set their style
                if (idx == 0)   elem.style.height = "20px";                  // the first one should stay at the top
                else {
                    elem.style.height = "20%";                               // the other 5 are equally spread
                    elem.style.verticalAlign = 'bottom';                     // and stick the text at the bottom of cell
                }
            }).append("<td>").find("td");                                    // add td in each row and return them
    var step = (maxValue - minValue) / (legendLabels.length - 1);            // -1 because it includes min and max
    legendLabels.each(function(idx, elem) {
        elem.innerHTML = (maxValue - idx * step).toFixed(3)
    })
}
// ================================= LEGEND UPDATING FUNCTION  END   =================================

