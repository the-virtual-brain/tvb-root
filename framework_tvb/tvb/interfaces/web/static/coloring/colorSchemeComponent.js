var _colorSchemeColors;
var _colorScale = null;                // the global color scale associated with this viewer
var _haveDefaultColorScheme = false;   // false before retrieving default colors from the server
var _refreshCallback = null ;          // this is called when color scheme changes update the visualiser

// ================================= COLOR SCHEME MODEL START =================================
/**
 * 3d viewer styling
 * Css class like idea. In the future we might move these to css
 * class -> prop value list
 */
var ColSchDarkTheme = {
    connectivityStepPlot : {
        lineColor: [0.1, 0.1, 0.2],
        noValueColor: [0.0, 0.0, 0.0],
        backgroundColor: [0.1, 0.1, 0.1, 1.0],
        outlineColor: [0.3, 0.3, 0.3],
        selectedOutlineColor: [0.2, 0.2, 0.8]
    },
    connectivityPlot : {
        backgroundColor: [0.1, 0.1, 0.1, 1.0]
    },
    surfaceViewer : {
        backgroundColor: [0.1, 0.1, 0.1, 1.0]
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
    rainbow: { theme: ColSchDarkTheme, _data_idx: 1},
    hotcold: { theme: ColSchDarkTheme, _data_idx: 2},
    TVB:     { theme: ColSchDarkTheme, _data_idx: 3},
    sparse:  { theme: ColSchDarkTheme, _data_idx: 4},

    RdYlBu      : { theme: ColSchDarkTheme, _data_idx: 13},
    Spectral    : { theme: ColSchDarkTheme, _data_idx: 14},
    YlGnBu      : { theme: ColSchDarkTheme, _data_idx: 15},
    RdPu        : { theme: ColSchDarkTheme, _data_idx: 16},
    Grays       : { theme: ColSchDarkTheme, _data_idx: 17},
    transparentRdYlBu: { theme: ColSchTransparentTheme, _data_idx: 13},

    matteo: { theme: ColSchDarkTheme, _data_idx: 23},
    cubehelix: { theme: ColSchDarkTheme, _data_idx: 24},
    termal: { theme: ColSchDarkTheme, _data_idx: 25},

    transparentJet: { theme: ColSchTransparentTheme, _data_idx: 1},
    transparentTVB: { theme: ColSchTransparentTheme, _data_idx: 3},
    transparentTermal: { theme: ColSchTransparentTheme, _data_idx: 25}
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
        scheme.muted_tex_v = (30 + 0.5)* bandHeight/textureSize;
        scheme.measurePoints_tex_v = 1.0;
    }
})();

/**
 * A color scale translates an activity to a color.
 * It does not have a GUI but it can be seen as the model of the color scheme component ui.
 * A color scheme object will contain a selected color scheme and the selected range and the number of bins etc
 * @constructor
 */
function ColorScale(minValue, maxValue, colorScheme, colorBins){
    if (minValue == null) {minValue = 0;}
    if (maxValue == null) {maxValue = 1;}
    if (colorScheme == null) {colorScheme = 'linear';}
    if (colorBins == null) {colorBins = 256;}
    this._minRange = minValue;       // the interest interval. Set by a slider in the ui
    this._maxRange = maxValue;
    this._minActivity = minValue;    // the full activity range
    this._maxActivity = maxValue;
    this._colorBins = colorBins;     // the number of discrete colors. Set by a slider in the ui
    this._colorScheme = colorScheme;
}

ColorScale.prototype.clampValue = function(value, min, max) {
    if (min == null) {min = 0;}
    if (max == null) {max = 1;}

    if (value > max) {
        return max;
    }
    if (value < min) {
        return min;
    }
    return value;
};

/**
 * Returns the current color scheme object
 */
ColorScale.prototype.colSchInfo = function(){
    return _ColSchemesInfo[this._colorScheme];
};

/**
 * Looks up an activity value in the current palette. Analog to the color scheme shader.
 * Takes into account the active range and number of bins.
 * @returns {number} the index in the palette corresponding to the activity 0..len(colors)
 */
ColorScale.prototype.getPaletteIndex = function(activity){
    activity = (activity - this._minRange)/(this._maxRange - this._minRange);
    // bin the activity
    activity  = Math.floor(activity  * this._colorBins) / this._colorBins;
    // We sample the interior of the array. If activity is between [0..1] we will obtain colors[1] and colors[254]
    // This linear transform implements it. The shader version does the same.
    activity = activity * 253.0/255.0 + 1.0/255.0;
    activity = this.clampValue(activity);
    // from 0..1 to 0..255 the color array range
    return Math.round(activity * 255); // nearest neighbour interpolation
};

ColorScale.prototype.getBounds = function() {
    return { min: this._minRange, max:this._maxRange, bins: this._colorBins };
};

/**
 * Looks up an activity value in the current palette. Analog to the color scheme shader.
 * Takes into account the active range and number of bins.
 * @returns {[number]} rgb in 0..1 units
 */
ColorScale.prototype.getColor = function(activity){
    // The color array for the current scheme
    var colors = _colorSchemeColors[this.colSchInfo()._data_idx];
    var col = colors[this.getPaletteIndex(activity)];
    // this function returns float colors
    return [col[0]/255, col[1]/255, col[2]/255];
};

ColorScale.prototype.getCssColor = function(activity) {
    var rgb_values = this.getColor(activity);
    return "rgb(" + Math.round(rgb_values[0]*255) + "," + Math.round(rgb_values[1]*255) + "," + Math.round(rgb_values[2]*255) + ")";
};

/**
 * Returns a color for the given point in interval (min, max), according to the current <code>_colorScheme</code>
 * Values outside the interval are clamped
 *
 * @param pointValue The value whose corresponding color is returned
 * @param max Upper bound for pointValue
 * @param min Lower bound for pointValue
 */
ColorScale.prototype.getGradientColor = function(pointValue, min, max){
    // The color array for the current scheme
    var colors = _colorSchemeColors[this.colSchInfo()._data_idx];

    if (min === max) {         // the interval is empty, so start color is the only possible one
        var col = colors[0];
        return [col[0]/255, col[1]/255, col[2]/255];
    }
    pointValue = this.clampValue(pointValue, min, max); // avoid rounding problems

    // As we are given explicit bounds we have to rescale the value to an activity
    var normalizedValue = (pointValue - min) / (max - min);  // to 0..1
    var activity = this._minActivity + normalizedValue * (this._maxActivity - this._minActivity); // to _minActivity.._maxActivity

    return this.getColor(activity);
};

// ================================= COLOR SCHEME MODEL END =================================


// ================================= COLOR SCHEME CONTROLLER START =================================

/**
 * Returns the current color scheme object
 */
function ColSchInfo(){
    return _ColSchemesInfo[_colorScale._colorScheme];
}

/**
 * For each color scheme return a 3d theme
 */
function ColSchGetTheme(){
    return ColSchInfo().theme;
}

function ColSchGetBounds(){
    return _colorScale.getBounds();
}

/**
 * Creates a tiled color picker with 12 colors: 10 from the spectral color scheme, a black and a white
 * @param selector Selector of the <nav> element of this color picker
 */
function ColSchCreateTiledColorPicker(selector){
    var N = 10;
    var colors = _colorSchemeColors[_ColSchemesInfo.Spectral._data_idx];
    var tiles = [[255, 255, 255], [5, 5, 5]];
    for (var i = 0; i < N; i++){
        tiles.push(colors[1 + Math.floor(i / N * 254)]);
    }
     return new TVBUI.ColorTilePicker(selector, tiles);
}


/**
 * Sets the current color scheme to the given one
 * @param scheme The name of the color scheme. See _ColSchemesInfo for supported schemes.
 * @param notify When true, trigger an ajax call to store on the server the changed setting. It will also notify listeners.
 */
function ColSch_setColorScheme(scheme, notify) {
    _colorScale._colorScheme = scheme;
    if(notify){
        //could throttle this
        doAjaxCall({
            url: '/user/set_viewer_color_scheme/' + scheme
        });
        if (_refreshCallback) {
            _refreshCallback();
        }
    } else {
        $("#setColorSchemeSelId").val(scheme);
    }
}
/**
 * Initializes the color scheme component.
 * The parameters are used to rescale a value to 0..1
 * Bounds are optional, will default to 0..1
 * @param [minValue] activity min
 * @param [maxValue] activity max
 */
function ColSch_initColorSchemeComponent(minValue, maxValue){
    // This will get called many times for a page for historical reasons
    // So we have to initialize what is not initialized and update what is already initialized

    if (!_haveDefaultColorScheme) {
        _colorScale = new ColorScale(minValue, maxValue);
    } else { // not first call, propagate the color scheme
        _colorScale = new ColorScale(minValue, maxValue, _colorScale._colorScheme);
    }

    if(!_colorSchemeColors) {
        doAjaxCall({
            url: '/user/get_color_schemes_json',
            type: 'GET',
            async: false,
            success: function (data) {
                _colorSchemeColors = JSON.parse(data);
            }
        });
    }
    if (!_haveDefaultColorScheme) { // on very first call, set the default color scheme
        doAjaxCall({
            url:'/user/get_viewer_color_scheme',
            async: false,
            success:function(data){
                ColSch_setColorScheme(JSON.parse(data), false);
                _haveDefaultColorScheme = true;
            }
        });
    }
}

/**
 * Initialises the settings for all color schemes
 *
 * @param minValue The minimum value for the linear slider
 * @param maxValue The maximum value for the linear slider
 * @param [refreshFunction] A reference to the function which updates the visualiser
 */
function ColSch_initColorSchemeGUI(minValue, maxValue, refreshFunction) {
    ColSch_initColorSchemeComponent(minValue, maxValue);
    _refreshCallback = refreshFunction;
    var elemSliderSelector = $("#rangerForLinearColSch");
    var elemMin = $("#sliderMinValue");
    var elemMax = $("#sliderMaxValue");
    var elemColorNoSlider = $("#sliderForSparseColSch");
    var elemColorNo = $("#ColSch_colorNo");

    // initialise the range UI
    elemSliderSelector.slider({
        range: true, min: minValue, max: maxValue, step: (maxValue - minValue) / 1000, // 1000 steps between max and min
        values: [minValue, maxValue],
        slide: function(event, ui) {
            elemMin.html(ui.values[0].toFixed(3));
            elemMax.html(ui.values[1].toFixed(3));
        },
        change: function(event, ui) {
            _colorScale._minRange = ui.values[0];
            _colorScale._maxRange = ui.values[1];
            if (_refreshCallback) { _refreshCallback(); }
        }
    });
    elemMin.html(minValue.toFixed(3));
    elemMax.html(maxValue.toFixed(3));
    // initialise the number of colors UI
    elemColorNo.html(_colorScale._colorBins);
    elemColorNoSlider.slider({ // and exponential slider for number of color bins
        min: 1, max: 8, step: 1, value: 8,
        slide: function (event, ui) {
            var nbins = Math.pow(2, ui.value);
            elemColorNo.html(nbins);
        },
        change: function (event, ui) {
            _colorScale._colorBins = Math.pow(2, ui.value);
            if (_refreshCallback) { _refreshCallback(); }
        }
    });
}

/**
 * Returns a color for the given point in interval (min, max), according to the current <code>_colorScheme</code>
 * Values outside the interval are clamped
 *
 * @param pointValue The value whose corresponding color is returned
 * @param max Upper bound for pointValue
 * @param min Lower bound for pointValue
 */
function getGradientColor(pointValue, min, max) {
    return _colorScale.getGradientColor(pointValue, min, max);
}

/**
 * Looks up an activity value in the current palette. Analog to the color scheme shader.
 * Takes into account the active range and number of bins.
 * @returns {[number]} rgb in 0..1 units
 */
function ColSch_getColor(activity){
    return _colorScale.getColor(activity);
}

function ColSch_getAbsoluteGradientColorString(pointValue) {
    return _colorScale.getCssColor(pointValue);
}

function ColSch_getGradientColorString(pointValue, min, max) {
    var rgb_values = getGradientColor(pointValue, min, max);
    return "rgb(" + Math.round(rgb_values[0]*255) + "," + Math.round(rgb_values[1]*255) + "," + Math.round(rgb_values[2]*255) + ")";
}

// ================================= COLOR SCHEME CONTROLLER  END  =================================

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
        elem.innerHTML = round_number(maxValue - idx * step, 3);
    });
}

// @private
function round_number(num, dec) {
    return Math.floor(num * Math.pow(10, dec)) / Math.pow(10, dec);
}
// ================================= LEGEND UPDATING FUNCTION  END   =================================

