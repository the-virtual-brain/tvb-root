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
 * .. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
 * .. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
 **/

// color scheme controller
var ColSch = {
    /**
     * The global color scale. This is kept in synch with the GUI
     * @type ColorScale
     */
    colorScale: null,
    _colorSchemeColors: null,           // an array mapping a color scheme index to its color array
    _haveDefaultColorScheme: false,     // false before retrieving default colors from the server
    _refreshCallback: null              // this is called when color scheme changes update the visualiser
};

// ================================= COLOR SCHEME MODEL START =================================
/**
 * 3d viewer styling
 * Css class like idea. In the future we might move these to css
 * class -> prop value list
 */
ColSch.DarkTheme = {
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

ColSch.LightTheme = {
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

ColSch.TransparentTheme = {
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
 *
 * @param theme a theme object containing colors of various objects
 * @param data_idx the index in _colorSchemeColors of the theme
 * @constructor
 */
function ColorScheme(theme, data_idx){
    var bandHeight = 8;
    var textureSize = 256;

    this.theme = theme;                                     // a theme object containing colors of various objects
    this.tex_v = (data_idx + 0.5) * bandHeight/textureSize; // the v texture coordinate of the color scheme
    this.muted_tex_v = (30 + 0.5)* bandHeight/textureSize;  // the v texture coordinate for the scheme used to paint deselected regions
    this.measurePoints_tex_v = 1.0;                         // the v texture coordinate for the scheme used to paint measure points
    this._data_idx = data_idx;                              // the index in _colorSchemeColors of the theme
}

/**
 * A table of color scheme objects
 */
ColSch.schemes = {
    linear:  new ColorScheme(ColSch.DarkTheme, 0),
    rainbow: new ColorScheme(ColSch.DarkTheme, 1),
    hotcold: new ColorScheme(ColSch.DarkTheme, 2),
    TVB:     new ColorScheme(ColSch.DarkTheme, 3),
    sparse:  new ColorScheme(ColSch.DarkTheme, 4),

    RdYlBu:   new ColorScheme(ColSch.DarkTheme, 13),
    Spectral: new ColorScheme(ColSch.DarkTheme, 14),
    YlGnBu:   new ColorScheme(ColSch.DarkTheme, 15),
    RdPu:     new ColorScheme(ColSch.DarkTheme, 16),
    Grays:    new ColorScheme(ColSch.DarkTheme, 17),

    matteo:    new ColorScheme(ColSch.DarkTheme, 23),
    cubehelix: new ColorScheme(ColSch.DarkTheme, 24),
    termal:    new ColorScheme(ColSch.DarkTheme, 25),

    whiteJet:  new ColorScheme(ColSch.LightTheme, 1),
    whiteTVB:  new ColorScheme(ColSch.LightTheme, 3),

    transparentJet:    new ColorScheme(ColSch.TransparentTheme, 1),
    transparentTVB:    new ColorScheme(ColSch.TransparentTheme, 3),
    transparentRdYlBu: new ColorScheme(ColSch.TransparentTheme, 13),
    transparentSparse: new ColorScheme(ColSch.TransparentTheme, 19),
    transparentMatteo: new ColorScheme(ColSch.TransparentTheme, 20),
    transparentTermal: new ColorScheme(ColSch.TransparentTheme, 25)
};


/**
 * A color scale translates an activity to a color.
 * It does not have a GUI but it can be seen as the model of the color scheme component ui.
 * A color scheme object will contain a selected color scheme and the selected range and the number of bins etc
 * @constructor
 */
function ColorScale(minValue, maxValue, colorSchemeName, colorBins, centralHoleDiameter){
    if (minValue == null) {minValue = 0;}
    if (maxValue == null) {maxValue = 1;}
    if (colorSchemeName == null) {colorSchemeName = 'linear';}
    if (colorBins == null) {colorBins = 256;}
    this._minRange = minValue;       // the interest interval. Set by a slider in the ui
    this._maxRange = maxValue;
    this._minActivity = minValue;    // the full activity range
    this._maxActivity = maxValue;
    this._colorBins = colorBins;     // the number of discrete colors. Set by a slider in the ui
    this._centralHoleDiameter = centralHoleDiameter || 0;   // range 0..1
    this._colorSchemeName = colorSchemeName;
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
ColorScale.prototype.getColorScheme = function(){
    return ColSch.schemes[this._colorSchemeName];
};

/**
 * Looks up an activity value in the current palette. Analog to the color scheme shader.
 * Takes into account the active range and number of bins.
 * @returns {number} the index in the palette corresponding to the activity 0..len(colors)
 */
ColorScale.prototype.getPaletteIndex = function(activity){
    // to 0..1
    activity = (activity - this._minRange)/(this._maxRange - this._minRange);
    // treat central values as out of range
    if ( Math.abs(activity - 0.5) < this._centralHoleDiameter / 2 ){
        return 0;
    }
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
    return {
        min: this._minRange, max:this._maxRange,
        bins: this._colorBins, centralHoleDiameter: this._centralHoleDiameter
    };
};

/**
 * Looks up an activity value in the current palette. Analog to the color scheme shader.
 * Takes into account the active range and number of bins.
 * @returns {[number]} rgba in 0..1 units
 */
ColorScale.prototype.getColor = function(activity){
    // The color array for the current scheme
    var colors = ColSch._colorSchemeColors[this.getColorScheme()._data_idx];
    var col = colors[this.getPaletteIndex(activity)];
    // this function returns float colors
    return [col[0]/255, col[1]/255, col[2]/255, 1.0];
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
    var colors = ColSch._colorSchemeColors[this.getColorScheme()._data_idx];

    if (min === max) {         // the interval is empty, so start color is the only possible one
        var col = colors[0];
        return [col[0]/255, col[1]/255, col[2]/255, 1.0];
    }
    pointValue = this.clampValue(pointValue, min, max); // avoid rounding problems

    // As we are given explicit bounds we have to rescale the value to an activity
    var normalizedValue = (pointValue - min) / (max - min);  // to 0..1
    var activity = this._minActivity + normalizedValue * (this._maxActivity - this._minActivity); // to _minActivity.._maxActivity

    return this.getColor(activity);
};

ColorScale.prototype._toCss = function(rgb_values){
    return "rgba(" + Math.round(rgb_values[0] * 255) + "," + Math.round(rgb_values[1] * 255) + "," +
                     Math.round(rgb_values[2] * 255) + "," + Math.round(rgb_values[3] * 255) + ")";
};

/** @see getColor */
ColorScale.prototype.getCssColor = function(activity) {
    return this._toCss(this.getColor(activity));
};

/** @see getGradientColor */
ColorScale.prototype.getCssGradientColor = function(pointValue, min, max) {
    return this._toCss(this.getGradientColor(pointValue, min, max));
};

/**
 * Subclass of ColorScale that returns out of scale values transparent
 * NOTE: subclassing ColorScales does not work with the shader based color schemes.
 * @constructor
 * @extends RegionSelectComponent
 */
function AlphaClampColorScale(minValue, maxValue, colorSchemeName, colorBins, centralHoleDiameter, alpha){
    ColorScale.call(this, minValue, maxValue, colorSchemeName, colorBins, centralHoleDiameter);
    this.alpha = alpha;
}

// proto chain setup.
AlphaClampColorScale.prototype = Object.create(ColorScale.prototype);

AlphaClampColorScale.prototype.getColor = function(activity){
    // The color array for the current scheme
    var colors = ColSch._colorSchemeColors[this.getColorScheme()._data_idx];
    var pIdx = this.getPaletteIndex(activity);
    if (pIdx === 0 || pIdx === 255){
        return [0, 0, 0, 0];
    }
    var col = colors[pIdx];
    // this function returns float colors
    return [col[0]/255, col[1]/255, col[2]/255, this.alpha];
};

// ================================= COLOR SCHEME MODEL END =================================


// ================================= COLOR SCHEME CONTROLLER START =================================

/**
 * Returns the current color scheme object
 */
function ColSchInfo(){
    return ColSch.colorScale.getColorScheme();
}

/**
 * For each color scheme return a 3d theme
 */
function ColSchGetTheme(){
    return ColSch.colorScale.getColorScheme().theme;
}

function ColSchGetBounds(){
    return ColSch.colorScale.getBounds();
}

/**
 * Creates a tiled color picker with 12 colors: 10 from the spectral color scheme, a black and a white
 * @param selector Selector of the <nav> element of this color picker
 */
function ColSchCreateTiledColorPicker(selector){
    var N = 10;
    var colors = ColSch._colorSchemeColors[ColSch.schemes.Spectral._data_idx];
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
    ColSch.colorScale._colorSchemeName = scheme;
    if(notify){
        //could throttle this
        doAjaxCall({
            url: '/user/set_viewer_color_scheme/' + scheme
        });
        if (ColSch._refreshCallback) {
            ColSch._refreshCallback();
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

    if (!ColSch._haveDefaultColorScheme) {
        ColSch.colorScale = new ColorScale(minValue, maxValue);
    } else if (minValue != null && maxValue != null){ // not first call, update min and max if provided
        ColSch.colorScale = new ColorScale(minValue, maxValue, ColSch.colorScale._colorSchemeName);
    }

    if(!ColSch._colorSchemeColors) {
        doAjaxCall({
            url: '/user/get_color_schemes_json',
            type: 'GET',
            async: false,
            success: function (data) {
                ColSch._colorSchemeColors = JSON.parse(data);
            }
        });
    }
    if (!ColSch._haveDefaultColorScheme) { // on very first call, set the default color scheme
        doAjaxCall({
            url:'/user/get_viewer_color_scheme',
            async: false,
            success:function(data){
                ColSch_setColorScheme(JSON.parse(data), false);
                ColSch._haveDefaultColorScheme = true;
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
    ColSch._refreshCallback = refreshFunction;
    var elemSliderSelector = $("#rangerForLinearColSch");
    var elemMin = $("#sliderMinValue");
    var elemMax = $("#sliderMaxValue");
    var elemColorNoSlider = $("#sliderForSparseColSch");
    var elemColorNo = $("#ColSch_colorNo");
    var elemSliderMiddleHole = $('#rangerForMiddleHoleDiameter');

    // initialise the range UI
    if (isNaN(minValue) || isNaN(maxValue)) {
        elemSliderSelector.slider({ disabled: true });
    } else {
        elemSliderSelector.slider({ disabled: false });
    }

    elemSliderSelector.slider({
        range: true, min: minValue, max: maxValue, step: (maxValue - minValue) / 1000, // 1000 steps between max and min
        values: [minValue, maxValue],
        slide: function(event, ui) {
            elemMin.html(ui.values[0].toFixed(3));
            elemMax.html(ui.values[1].toFixed(3));
        },
        change: function(event, ui) {
            if (!isNaN(ui.value)) {
                ColSch.colorScale._minRange = ui.values[0];
                ColSch.colorScale._maxRange = ui.values[1];
            }
            if (ColSch._refreshCallback) { ColSch._refreshCallback(); }
        }
    });
    elemMin.html(minValue.toFixed(3));
    elemMax.html(maxValue.toFixed(3));
    // initialise the number of colors UI
    elemColorNo.html(ColSch.colorScale._colorBins);
    elemColorNoSlider.slider({ // and exponential slider for number of color bins
        min: 1, max: 8, step: 1, value: 8,
        slide: function (event, ui) {
            var nbins = Math.pow(2, ui.value);
            elemColorNo.html(nbins);
        },
        change: function (event, ui) {
            ColSch.colorScale._colorBins = Math.pow(2, ui.value);
            if (ColSch._refreshCallback) { ColSch._refreshCallback(); }
        }
    });
    // initialize the near mean trimming slider
    elemSliderMiddleHole.slider({
        min: 0, max: 1, step: 1/128, value: 0,
        change: function (event, ui) {
            ColSch.colorScale._centralHoleDiameter = ui.value;
            if (ColSch._refreshCallback) { ColSch._refreshCallback(); }
        }
    });
}

// todo: consider inlining these thin wrappers

/**
 * @see ColorScale.getGradientColor
 */
function getGradientColor(pointValue, min, max) {
    return ColSch.colorScale.getGradientColor(pointValue, min, max);
}

/**
 * @see ColorScale.getColor
 */
function ColSch_getColor(activity){
    return ColSch.colorScale.getColor(activity);
}

/**
 * @see ColorScale.getCssColor
 */
function ColSch_getAbsoluteGradientColorString(pointValue) {
    return ColSch.colorScale.getCssColor(pointValue);
}

/**
 * @see ColorScale.getCssGradientColor
 */
function ColSch_getGradientColorString(pointValue, min, max) {
    return ColSch.colorScale.getCssGradientColor(pointValue, min, max);
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
    function round_number(num, dec) {
        return Math.floor(num * Math.pow(10, dec)) / Math.pow(10, dec);
    }

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

// ================================= LEGEND UPDATING FUNCTION  END   =================================

