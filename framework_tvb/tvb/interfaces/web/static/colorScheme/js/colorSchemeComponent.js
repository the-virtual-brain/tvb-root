var nodeColorRGB = [255, 255, 255];
var _colorSchemeColors;
var _colorScheme = null;                // the color scheme to be used for current drawing
var _minActiv, _maxActiv;               // keep the interest interval
var _refreshCallback = null ;           // this is called when color scheme changes update the visualiser
var _sparseColorNo = 256;

// ================================= COLOR SCHEME STRUCTURES START =================================
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
        backgroundColor: [0.05, 0.05, 0.05, 1.0]
        //, boundaryLineColor
        //, navigatorColor
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
        backgroundColor: [1.0, 1.0, 1.0, 1.0]
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
        backgroundColor: [0.0, 0.0, 0.0, 0.0]
    }
};

/**
 * A table of color scheme objects
 * Fields:
 *     theme: a theme object containing colors of various objects
 *     tex_v: the v texture coordinate of the color scheme
 *     muted_tex_v: the v texture coordinate for the scheme used to paint deselected regions
 *     measurePoints_tex_v: the v texture coordinate for the scheme used to paint measure points
 *     _data_idx: the index in _colorSchemeColors of the theme
 */
var _ColSchemesInfo = {
    linear:  { theme: ColSchDarkTheme, _data_idx: 0},
    TVB:     { theme: ColSchDarkTheme, _data_idx: 3},
    rainbow: { theme: ColSchDarkTheme, _data_idx: 1},
    hotcold: { theme: ColSchDarkTheme, _data_idx: 2},
    sparse:  { theme: ColSchDarkTheme, _data_idx: 4},
    lightHotcold:   { theme: ColSchLightTheme, _data_idx: 2},
    lightTVB:       { theme: ColSchLightTheme, _data_idx: 3},
    transparentHotCold: { theme: ColSchTransparentTheme, _data_idx: 2},
    marteli: { theme: ColSchDarkTheme, _data_idx: 10},
    cubehelix: { theme: ColSchDarkTheme, _data_idx: 11},
    termal: { theme: ColSchDarkTheme, _data_idx: 12},
    brewer1: { theme: ColSchDarkTheme, _data_idx: 9}
};

// Add texture v coordinates to _ColSchemesInfo based on the _data_idx
// Auto executed function so we do not pollute globals
(function() {
    var bandHeight = 8;
    var textureSize = 256;
    for (var n in _ColSchemesInfo) {
        // band indices are the same as the indices in the _colorSchemeColors
        var scheme = _ColSchemesInfo[n];
        scheme.tex_v = (scheme._data_idx + 0.5) * bandHeight/textureSize;
        scheme.muted_tex_v = 0.5;
        scheme.measurePoints_tex_v = 1.0;
    }
})();

/**
 * Returns the current color scheme object
 */
function ColSchInfo(){
    return _ColSchemesInfo[_colorScheme||'linear'];
}

/**
 * For each color scheme return a 3d theme
 */
function ColSchGetTheme(){
    return ColSchInfo().theme;
}

function ColSchGetBounds(){
    return { min: _minActiv, max:_maxActiv, bins: _sparseColorNo };
}
// ================================= COLOR SCHEME STRUCTURES END =================================


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

function clampValue(value, min, max) {
    if (min == null) {min = 0;}
    if (max == null) {max = 1;}

    if (value > max) {
        return max;
    }
    if (value < min) {
        return min;
    }
    return value;
}

// ================================= COLOR SCHEME FUNCTIONS START =================================

/**
 * Sets the current color scheme to the given one
 * @param scheme The name of the color scheme. See _ColSchemesInfo for supported schemes.
 * @param notify_server When TRUE, trigger an ajax call to store on the server changed setting.
 */
function ColSch_setColorScheme(scheme, notify_server) {
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

function ColSch_initColorSchemeComponent(){
    if(_colorSchemeColors != null){
        return;
    }
    doAjaxCall({
        url: '/user/get_color_schemes_json',
        async: false,
        success: function(data){
            _colorSchemeColors = JSON.parse(data);
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
            _minActiv = ui.values[0];
            _maxActiv = ui.values[1];
            if (_refreshCallback) { _refreshCallback(); }
        }
    });
    $("#sliderMinValue").html(minValue.toFixed(3));
    $("#sliderMaxValue").html(maxValue.toFixed(3));
    _minActiv = minValue;            // on start the whole interval is selected
    _maxActiv = maxValue;

    // initialise the sparse params
    var colorNoUIElem = $("#ColSch_colorNo");                    // cache the jQuery selector
    colorNoUIElem.html(_sparseColorNo);
    $("#sliderForSparseColSch").slider({
        min: 1, max: 8, step: 1, value: 8,
        slide: function (event, ui) {
            var nbins = Math.pow(2, ui.value);
            colorNoUIElem.html(nbins);
        },
        change: function (event, ui) {
            _sparseColorNo = Math.pow(2, ui.value);
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
    // The color array for the current scheme
    var colors = _colorSchemeColors[ColSchInfo()._data_idx];

    if (min == max) {         // the interval is empty, so start color is the only possible one
        return colors[0];
    }
    pointValue = clampValue(pointValue, min, max); // avoid rounding problems

    //scale activity within given range to [0,1]
    var normalizedValue = (pointValue - min) / (max - min);
     // bin the activity
    normalizedValue  = Math.floor(normalizedValue  * _sparseColorNo) / _sparseColorNo;
    // We sample the interior of the array. If normalizedValue is between [0..1] we will obtain colors[1] and colors[254]
    // This linear transform implements it. The shader version does the same.
    normalizedValue = normalizedValue * 253.0/255.0 + 1.0/255.0;
    normalizedValue = clampValue(normalizedValue);
    // from 0..1 to 0..255 the color array range
    var idx = Math.round(normalizedValue * 255); // nearest neighbour interpolation
    var col = colors[idx];
    // this function returns float colors
    return [col[0]/255, col[1]/255, col[2]/255];
}

function ColSch_getGradientColorString(pointValue, min, max) {
    var rgb_values = getGradientColor(pointValue, min, max);
    return "rgb(" + Math.round(rgb_values[0]*255) + "," + Math.round(rgb_values[1]*255) + "," + Math.round(rgb_values[2]*255) + ")";
}

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
        if (outputArray) {
            outputArray.set(color, i * 4);
        } else {
            result.concat(color);
        }
    }
    return result;
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
    if (!table.length) {
        table = $("<table>").appendTo(container);                            // create one if it doesn't exist
    }
    table.height(height);                                                    // set its height
    var legendLabels = $(table).find("td");                                  // search for td
    if (!legendLabels.length) {                                               // if none is found, assume tr also don't exist
        legendLabels = $(table).append("<tr><tr><tr><tr><tr><tr>")          // so add 6 rows
            .find("tr").each(function (idx, elem) {                          // get them and set their style
                if (idx === 0) {                                            // the first one should stay at the top
                    elem.style.height = "20px";
                } else {
                    elem.style.height = "20%";                               // the other 5 are equally spread
                    elem.style.verticalAlign = 'bottom';                     // and stick the text at the bottom of cell
                }
            }).append("<td>").find("td");                                    // add td in each row and return them
    }
    var step = (maxValue - minValue) / (legendLabels.length - 1);            // -1 because it includes min and max
    legendLabels.each(function(idx, elem) {
        elem.innerHTML = (maxValue - idx * step).toFixed(3)
    });
}
// ================================= LEGEND UPDATING FUNCTION  END   =================================

