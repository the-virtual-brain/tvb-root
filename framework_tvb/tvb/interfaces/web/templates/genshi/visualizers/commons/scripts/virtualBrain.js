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

/* The comment below lists the global functions used in this file.
 * It is here to make jshint happy and to document these implicit global dependencies.
 * In the future we might group these into namespace objects.
 * ( Global state is not in this list except gl; let them be warnings )
 */

/* globals gl, GL_shaderProgram, displayMessage, HLPR_readJSONfromFile, readDataPageURL,
    GL_handleKeyDown, GL_handleKeyUp, GL_handleMouseMove, GL_handleMouseWeel,
    initGL, updateGLCanvasSize, LEG_updateLegendVerticesBuffers,
    basicInitShaders, basicInitSurfaceLighting, GL_initColorPickFrameBuffer,
    ColSch_loadInitialColorScheme, ColSchGetTheme, LEG_generateLegendBuffers, LEG_initMinMax
    */

/**
 * WebGL methods "inheriting" from webGL_xx.js in static/js.
 */

var BRAIN_CANVAS_ID = "GLcanvas";
/**
 * Variables for displaying Time and computing Frames/Sec
 */
var lastTime = 0;
var framestime = [50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
                  50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50];

/**
 * Time like entities:
 * The movie time
 *      Measured in 'time steps'
 *      An index in the activitiesData array
 * The display time
 *      Measured in 'ticks'
 *      Updated every TICK_STEP ms.
 *      We do not keep the value of this time.
 * The displayed movie time
 *      The int value of it is in currentTimeValue.
 *      Measured in 'time steps'.
 *      Synchronizes the movie time to the display time.
 */

/**
 * Granularity of the display time in ms.
 */
var TICK_STEP = 33; // 30Hz
/**
 * How many movie time steps for a display tick.
 * If this is < 1 a movie frame will last 1/timeStepsPerTick ticks
 */
var timeStepsPerTick = 1;
/**
 * The integer part of timeStepsPerTick
 */
var TIME_STEP = 1;
/**
 * The current time in the activity movie.
 * An index of the current movie frame.
 * When timeStepsPerTick > it increments by TIME_STEP every tick.
 * When timeStepsPerTick < 1 it increments by 1 every 1/timeStepsPerTick tick.
 */
var currentTimeValue = 0;
/**
 * The maximum possible value of currentTimeValue
 */
var MAX_TIME = 0;
/**
 * For how many display ticks have we drawn the same time step.
 */
var elapsedTicksPerTimeStep = 0;
/**
 * At the maximum speed the time line finishes in 32 steps
 * This is approximately 1s wall time (ignoring data fetches).
 */
var ACTIVITY_FRAMES_IN_TIME_LINE_AT_MAX_SPEED = 32;

var sliderSel = false;

var isPreview = false;

var brainBuffers = [];
var brainLinesBuffers = [];
var shelfBuffers = [];
var measurePointsBuffers = [];
/**
 * This buffer arrays will contain:
 * arr[i][0] Vertices buffer
 * arr[i][1] Normals buffer
 * arr[i][2] Triangles indices buffer
 * arr[i][3] Color buffer (same length as vertices /3 * 4) in case of one-to-one mapping
 * arr[i][3] not used
 * arr[i][4] Alpha Indices Buffer Indices of the 3 closest measurement points, in care of not one-to-one mapping
 */

var boundaryVertexBuffers = [];
var boundaryNormalsBuffers = [];
var boundaryEdgesBuffers = [];

var activitiesData = [], timeData = [], measurePoints = [], measurePointsLabels = [];

var pageSize = 0;
var urlBase = '';
var selectedMode = 0;
var selectedStateVar = 0;
var currentActivitiesFileLength = 0;
var nextActivitiesFileData = [];
var totalPassedActivitiesData = 0;
var shouldIncrementTime = true;
var currentAsyncCall = null;

var NO_OF_MEASURE_POINTS = 0;
var NEXT_PAGE_THREASHOLD = 100;

var activityMin = 0, activityMax = 0;
var isOneToOneMapping = false;
var isDoubleView = false;
var isEEGView = false;
var drawingMode;
var VS_showLegend = true;
var isInternalSensorView = false;
var displayMeasureNodes = false;
var isFaceToDisplay = false;

var drawNavigator = false;
var drawTriangleLines = false;
var drawBoundaries = false;

var drawSpeculars = false;
/**
 * Used to determine which buffer chunks belong to a hemisphere.
 * The chunks are used to limit geometry size for a draw call.
 */
var VS_hemisphere_chunk_mask = null;
var bufferSetsMask = null;
var VS_hemisphereVisibility = null;
/**
 * What regions are selected to be shown.
 * Unselected regions  are greyed out.
 * This is used only by the brain activity movie for region level activity.
 * For static viewers it is initialized to a full selection
 */
var VS_selectedRegions = [];
/**
 * camera settings
 */
var near = 0.1;

// index of the currently selected node. This is equivalent to CONN_pickedIndex
var VS_pickedIndex = -1;


var VB_BrainNavigator;


function VS_init_hemisphere_mask(hemisphere_chunk_mask){
    VS_hemisphere_chunk_mask = hemisphere_chunk_mask;
    if (hemisphere_chunk_mask != null){
        bufferSetsMask = [];
        for(var i = 0; i < VS_hemisphere_chunk_mask.length; i++){
            bufferSetsMask[i] = 1;
        }
    }
}

function VS_SetHemisphere(h){
    VS_hemisphereVisibility = h;
    for(var i = 0; i < VS_hemisphere_chunk_mask.length; i++){
        if ( h == null ){
            bufferSetsMask[i] = 1;
        }else if (h === 'l'){
            bufferSetsMask[i] = 1 - VS_hemisphere_chunk_mask[i];
        }else if (h === 'r'){
            bufferSetsMask[i] = VS_hemisphere_chunk_mask[i];
        }
    }
}

function VS_StartPortletPreview(baseDatatypeURL, urlVerticesList, urlTrianglesList, urlNormalsList,
                                urlRegionMapList, minActivity, maxActivity, oneToOneMapping) {
    isPreview = true;
    pageSize = 1;
    urlBase = baseDatatypeURL;
    activitiesData = HLPR_readJSONfromFile(readDataPageURL(urlBase, 0, 1, selectedStateVar, selectedMode, TIME_STEP));
    if (oneToOneMapping === 'True') {
        isOneToOneMapping = true;
    }
    activityMin = parseFloat(minActivity);
    activityMax = parseFloat(maxActivity);
    var canvas = document.getElementById(BRAIN_CANVAS_ID);
    customInitGL(canvas);
    initShaders();
    if (urlVerticesList) {
        brainBuffers = initBuffers($.parseJSON(urlVerticesList), $.parseJSON(urlNormalsList), $.parseJSON(urlTrianglesList), 
                               $.parseJSON(urlRegionMapList), false);
    }
    ColSch_initColorSchemeComponent(false);
    LEG_generateLegendBuffers();
    VB_BrainNavigator = new NAV_BrainNavigator(isOneToOneMapping, brainBuffers, measurePoints, measurePointsLabels);
    
    var theme = ColSchGetTheme().surfaceViewer;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);

    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    // Enable keyboard and mouse interaction
    canvas.onkeydown = GL_handleKeyDown;
    canvas.onkeyup = GL_handleKeyUp;
    canvas.onmousedown = customMouseDown;
    canvas.oncontextmenu = function() { return false;};
    $(document).on('mousemove', GL_handleMouseMove);
    $(document).on('mouseup', customMouseUp);
    // We use drawScene instead of tick because tick's performance is worse.
    // Portlet previews are static, not movies. Tick's movie update is not required.
    // A call to updateColors has to be made to initialize the color buffer.
    updateColors(0);
    setInterval(drawScene, TICK_STEP);
}

function _VS_static_entrypoint(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                               noOfMeasurePoints, urlRegionMapList, urlMeasurePointsLabels,
                               boundaryURL, shelfObject, hemisphereChunkMask, showLegend, argDisplayMeasureNodes, argIsFaceToDisplay,
                               minMeasure, maxMeasure, urlMeasure){
    // initialize global configuration
    isDoubleView = false;
    isOneToOneMapping = false;
    shouldIncrementTime = false;
    AG_isStopped = true;
    VS_showLegend = showLegend;
    displayMeasureNodes = argDisplayMeasureNodes;
    isFaceToDisplay = argIsFaceToDisplay; // this could be retrieved from the dom like drawNavigator
    // make checkbox consistent with this flag
    $("#displayFaceChkId").attr('checked', isFaceToDisplay);
    drawNavigator = $("#showNavigator").prop('checked');
    // initialize global data
    var i;

    if (noOfMeasurePoints === 0){
        // we are viewing a surface with no region mapping
        // we mock 1 measure point
        measurePoints = [[0, 0, 0]];
        measurePointsLabels = [''];
        NO_OF_MEASURE_POINTS = 1;
        // mock one activity frame
        activityMin = 0;
        activityMax = 1;
        activitiesData = [[0]];
    } else {
        _initMeasurePoints(noOfMeasurePoints, urlMeasurePoints, urlMeasurePointsLabels);
        activityMin = parseFloat(minMeasure);
        activityMax = parseFloat(maxMeasure);
        var measure;
        if (urlMeasure === ''){
            // Empty url => The static viewer has to show a region map.
            // The measure will be a range(NO_OF_MEASURE_POINTS)
            measure = [];
            for(i = 0; i < NO_OF_MEASURE_POINTS; i++){
                measure.push(i);
            }
        } else {
            measure = HLPR_readJSONfromFile(urlMeasure);
        }
        // The activity data will contain just one frame containing the values of the connectivity measure.
        activitiesData = [measure];
    }

    for(i = 0; i < NO_OF_MEASURE_POINTS; i++){
        VS_selectedRegions.push(i);
    }

    var canvas = document.getElementById(BRAIN_CANVAS_ID);
    _initViewerGL(canvas, urlVerticesList, urlNormalsList, urlTrianglesList,
                  urlRegionMapList, urlLinesList, boundaryURL, shelfObject, hemisphereChunkMask);

    _bindEvents(canvas);

    //specify the re-draw function.
    if (_isValidActivityData()){
        setInterval(tick, TICK_STEP);
    }
}

function _VS_movie_entrypoint(baseDatatypeURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlRegionMapList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shelfObject, hemisphereChunkMask, urlMeasurePointsLabels, boundaryURL) {
    // initialize global configuration
    isDoubleView = doubleView;
    if (oneToOneMapping == 'True') {
        isOneToOneMapping = true;
    }
    // these global flags could be structured better
    isEEGView = isDoubleView && !isInternalSensorView;
    activityMin = parseFloat(minActivity);
    activityMax = parseFloat(maxActivity);
    pageSize = onePageSize;
    urlBase = baseDatatypeURL;

    // initialize global data
    _initMeasurePoints(noOfMeasurePoints, urlMeasurePoints, urlMeasurePointsLabels);
    _initTimeData(urlTimeList);
    initActivityData();

    if (isDoubleView) {
        $("#displayFaceChkId").trigger('click');
    }
    drawNavigator = $("#showNavigator").prop('checked');

    var canvas = document.getElementById(BRAIN_CANVAS_ID);

    _initViewerGL(canvas, urlVerticesList, urlNormalsList, urlTrianglesList,
                  urlRegionMapList, urlLinesList, boundaryURL, shelfObject, hemisphereChunkMask);

    _bindEvents(canvas);

    _initSliders();

    //specify the re-draw function.
    if (_isValidActivityData()){
        setInterval(tick, TICK_STEP);
    }
}

function _VS_init_cubicalMeasurePoints(){
    for (var i = 0; i < NO_OF_MEASURE_POINTS; i++) {
        measurePointsBuffers[i] = bufferAtPoint(measurePoints[i]);
    }
}

function VS_StartSurfaceViewer(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                               noOfMeasurePoints, urlRegionMapList, urlMeasurePointsLabels,
                               boundaryURL, shelveObject, minMeasure, maxMeasure, urlMeasure, hemisphereChunkMask){

    _VS_static_entrypoint(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                       noOfMeasurePoints, urlRegionMapList, urlMeasurePointsLabels,
                       boundaryURL, shelveObject, hemisphereChunkMask, false, false, false, minMeasure, maxMeasure, urlMeasure);
    _VS_init_cubicalMeasurePoints();
    ColSch_initColorSchemeParams(activityMin, activityMax);
}

function VS_StartEEGSensorViewer(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                               noOfMeasurePoints, urlMeasurePointsLabels,
                               shelfObject, minMeasure, maxMeasure, urlMeasure){
    isEEGView = true;
    _VS_static_entrypoint(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                               noOfMeasurePoints, '', urlMeasurePointsLabels,
                               '', shelfObject, null, false, true, true, minMeasure, maxMeasure, urlMeasure);
    _VS_init_cubicalMeasurePoints();
    if (urlVerticesList) {
        ColSch_initColorSchemeParams(activityMin, activityMax);
    }
}

function VS_StartBrainActivityViewer(baseDatatypeURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlRegionMapList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shelfObject, hemisphereChunkMask,
                    urlMeasurePointsLabels, boundaryURL, measurePointsSelectionGID) {
    _VS_movie_entrypoint(baseDatatypeURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                    urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                    urlRegionMapList, minActivity, maxActivity,
                    oneToOneMapping, doubleView, shelfObject, hemisphereChunkMask,
                    urlMeasurePointsLabels, boundaryURL);
    _VS_init_cubicalMeasurePoints();

    if (!isDoubleView){
        // If this is a brain activity viewer then we have to initialize the selection component
        _initChannelSelection(measurePointsSelectionGID);
        // For the double view the selection is the responsibility of the extended view functions
    }
}

function _isValidActivityData(){
    if(isOneToOneMapping){
        if(3 * activitiesData[0].length !== brainBuffers[0][0].numItems ){            
            displayMessage("The number of activity points should equal the number of surface vertices", "errorMessage");
            return false;
        }
    } else {
        if (NO_OF_MEASURE_POINTS !== activitiesData[0].length){
            displayMessage("The number of activity points should equal the number of regions", "errorMessage");
            return false;
        }
    }
    return true; 
}

/**
 * Scene setup common to all webgl brain viewers
 */
function _initViewerGL(canvas, urlVerticesList, urlNormalsList, urlTrianglesList,
                       urlRegionMapList, urlLinesList, boundaryURL, shelfObject, hemisphere_chunk_mask){
    customInitGL(canvas);
    GL_initColorPickFrameBuffer();
    initShaders();

    if(VS_showLegend){
        LEG_initMinMax(activityMin, activityMax);
        LEG_generateLegendBuffers();
    }

    if (urlVerticesList) {
        var parsedIndices = [];
        if (urlRegionMapList) {
            parsedIndices = $.parseJSON(urlRegionMapList);
        }
        brainBuffers = initBuffers($.parseJSON(urlVerticesList), $.parseJSON(urlNormalsList),
                                   $.parseJSON(urlTrianglesList), parsedIndices, isDoubleView);
    }

    VS_init_hemisphere_mask(hemisphere_chunk_mask);

    brainLinesBuffers = HLPR_getDataBuffers(gl, $.parseJSON(urlLinesList), isDoubleView, true);
    initRegionBoundaries(boundaryURL);
    
    if (shelfObject) {
        shelfObject = $.parseJSON(shelfObject);
        shelfBuffers = initBuffers(shelfObject[0], shelfObject[1], shelfObject[2], false, true);
    }

    VB_BrainNavigator = new NAV_BrainNavigator(isOneToOneMapping, brainBuffers, measurePoints, measurePointsLabels);

    var theme = ColSchGetTheme().surfaceViewer;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
}


function _bindEvents(canvas){
    // Enable keyboard and mouse interaction
    canvas.onkeydown = GL_handleKeyDown;
    canvas.onkeyup = GL_handleKeyUp;
    canvas.onmousedown = customMouseDown;
    $(document).on('mouseup', customMouseUp);
    $(canvas).on('contextmenu', _onContextMenu);
    document.onmousemove = GL_handleMouseMove;

    $(canvas).mousewheel(function(event, delta) {
        GL_handleMouseWeel(delta);
        return false; // prevent default
    });

    if (!isDoubleView) {
        var canvasX = document.getElementById('brain-x');
        if (canvasX) {
            canvasX.onmousedown = function (event) { VB_BrainNavigator.moveInXSection(event)};
        }
        var canvasY = document.getElementById('brain-y');
        if (canvasY) {
            canvasY.onmousedown = function (event) { VB_BrainNavigator.moveInYSection(event)};
        }
        var canvasZ = document.getElementById('brain-z');
        if (canvasZ) {
            canvasZ.onmousedown = function (event) { VB_BrainNavigator.moveInZSection(event)};
        }
    }
}

function _initMeasurePoints(noOfMeasurePoints, urlMeasurePoints, urlMeasurePointsLabels){
    if (noOfMeasurePoints > 0) {
        measurePoints = HLPR_readJSONfromFile(urlMeasurePoints);
        measurePointsLabels = HLPR_readJSONfromFile(urlMeasurePointsLabels);
        NO_OF_MEASURE_POINTS = measurePoints.length;
    } else {
        NO_OF_MEASURE_POINTS = 0;
        measurePoints = [];
        measurePointsLabels = [];
    }
}

function _initTimeData(urlTimeList){
    var timeUrls = $.parseJSON(urlTimeList);
    for (var i = 0; i < timeUrls.length; i++) {
        timeData = timeData.concat(HLPR_readJSONfromFile(timeUrls[i]));
    }
    MAX_TIME = timeData.length - 1;
}

function _updateSpeedSliderValue(stepsPerTick){
    var s;
    if (stepsPerTick >= 1){
        s = stepsPerTick.toFixed(0);
    }else{
        s = "1/" + (1/stepsPerTick).toFixed(0);
    }
    $("#slider-value").html(s);
}

function _initSliders(){
    var maxAllowedTimeStep = Math.ceil(MAX_TIME / ACTIVITY_FRAMES_IN_TIME_LINE_AT_MAX_SPEED);
    // after being converted to the exponential range maxSpeed must not exceed maxAllowedTimeStep
    var maxSpeedSlider = Math.min(10, 5 + Math.log(maxAllowedTimeStep)/Math.LN2);

    if (timeData.length > 0) {
        $("#sliderStep").slider({min:0, max: maxSpeedSlider, step: 1, value:5,
            stop: function() {
                refreshCurrentDataSlice();
                sliderSel = false;
            },
            slide: function(event, target) {
                // convert the linear 0..10 range to the exponential 1/32..1..32 range
                var newStep = Math.pow(2, target.value - 5);
                setTimeStep(newStep);
                _updateSpeedSliderValue(timeStepsPerTick);
                sliderSel = true;
            }
        });
        // Initialize slider for timeLine
        $("#slider").slider({ min:0, max: MAX_TIME,
            slide: function(event, target) {
                sliderSel = true;
                currentTimeValue = target.value;
                $('#TimeNow').val(currentTimeValue);
            },
            stop: function(event, target) {
                sliderSel = false;
                loadFromTimeStep(target.value);
            }
        });
    } else {
        $("#divForSliderSpeed").hide();
    }
    _updateSpeedSliderValue(timeStepsPerTick);

    $('#TimeNow').click(function(){
        if (!AG_isStopped){
            pauseMovie();
        }
        $(this).select();
    }).change(function(ev){
        var val = parseFloat(ev.target.value);
        if (val == null || val < 0 || val > MAX_TIME){
            val = 0;
            ev.target.value = 0;
        }
        $('#slider').slider('value', val);
        loadFromTimeStep(val);
    });
}

function _initChannelSelection(selectionGID){
    var vs_regionsSelector = TVBUI.regionSelector("#channelSelector", {filterGid: selectionGID});

    vs_regionsSelector.change(function(value){
        VS_selectedRegions = [];
        for(var i=0; i < value.length; i++){
            VS_selectedRegions.push(parseInt(value[i], 10));
        }
    });
    //sync region filter with initial selection
    VS_selectedRegions = [];
    var selection = vs_regionsSelector.val();
    for(var i=0; i < selection.length; i++){
        VS_selectedRegions.push(parseInt(selection[i], 10));
    }
    var mode_selector = TVBUI.modeAndStateSelector("#channelSelector", 0);
    mode_selector.modeChanged(VS_changeMode);
    mode_selector.stateVariableChanged(VS_changeStateVariable);
}

////////////////////////////////////////// GL Initializations //////////////////////////////////////////

function customInitGL(canvas) {
    window.onresize = function() {
        updateGLCanvasSize(BRAIN_CANVAS_ID);
        LEG_updateLegendVerticesBuffers();
    };
    initGL(canvas);
    drawingMode = gl.TRIANGLES;
    gl.newCanvasWidth = canvas.clientWidth;
    gl.newCanvasHeight = canvas.clientHeight;
    canvas.redrawFunctionRef = drawScene;            // interface-like function used in HiRes image exporting
    canvas.multipleImageExport = VS_multipleImageExport;
}

/** This callback handles image exporting from this canvas.*/
function VS_multipleImageExport(saveFigure){
    var canvas = this;

    function saveFrontBack(nameFront, nameBack){
        mvPushMatrix();
        // front
        canvas.drawForImageExport();
        saveFigure({suggestedName: nameFront});
        // back: rotate model around the vertical y axis in trackball space (almost camera space: camera has a z translation)
        var r  = createRotationMatrix(180, [0, 1, 0]);
        GL_mvMatrix = GL_cameraMatrix.x(r.x(GL_trackBallMatrix));
        canvas.drawForImageExport();
        saveFigure({suggestedName: nameBack});
        mvPopMatrix();
    }

    // using drawForImageExport because it handles resizing canvas for export
    // It is set on canvas in initGL and defers to drawscene.

    if (VS_hemisphere_chunk_mask != null) {    // we have 2 hemispheres
        if (VS_hemisphereVisibility == null) {  // both are visible => take them apart when taking picture
            VS_SetHemisphere('l');
            saveFrontBack('brain-LH-front', 'brain-LH-back');
            VS_SetHemisphere('r');
            saveFrontBack('brain-RH-front', 'brain-RH-back');
            VS_SetHemisphere(VS_hemisphereVisibility);
        } else if (VS_hemisphereVisibility === 'l') {  // LH is visible => take picture of it only
            saveFrontBack('brain-LH-front', 'brain-LH-back');
        } else if (VS_hemisphereVisibility === 'r') {
            saveFrontBack('brain-RH-front', 'brain-RH-back');
        }
    } else {
        // just save front-back view if no hemispheres
        saveFrontBack('brain-front', 'brain-back');
    }
}

function initShaders() {
    createAndUseShader("shader-fs", "shader-vs");
    if (isOneToOneMapping) {
        SHADING_Context.one_to_one_program_init(GL_shaderProgram);
    }else {
        SHADING_Context.region_progam_init(GL_shaderProgram, NO_OF_MEASURE_POINTS, legendGranularity);
    }
}

function setLighting(settings){
    settings = settings || {};
    var useVertexColors = settings.materialColor == null;
    gl.uniform1i(GL_shaderProgram.useVertexColors, useVertexColors);
    if (! useVertexColors){
        gl.uniform4fv(GL_shaderProgram.materialColor, settings.materialColor);
    }
    return basicSetLighting(settings);
}
///////////////////////////////////////~~~~~~~~START MOUSE RELATED CODE~~~~~~~~~~~//////////////////////////////////


function _onContextMenu(){
    if( !displayMeasureNodes || VS_pickedIndex === -1) {
        return false;
    }
    doPick = true;
    drawScene();
    $('#nodeNameId').text(measurePointsLabels[VS_pickedIndex]);
    $('#contextMenuDiv').css('left', event.offsetX).css('top', event.offsetY).show();
    return false;
}

var doPick = false;

function customMouseDown(event) {
    GL_handleMouseDown(event, $("#" + BRAIN_CANVAS_ID));
    $('#contextMenuDiv').hide();
    VB_BrainNavigator.temporaryDisableInTimeRefresh();
    if (displayMeasureNodes) {
        doPick = true;
    }
}

function customMouseUp(event) {
    GL_handleMouseUp(event);
    VB_BrainNavigator.endTemporaryDisableInTimeRefresh();
}

/////////////////////////////////////////~~~~~~~~END MOUSE RELATED CODE~~~~~~~~~~~//////////////////////////////////


////////////////////////////////////////~~~~~~~~~ WEB GL RELATED RENDERING ~~~~~~~/////////////////////////////////
/**
 * Update colors for all Positions on the brain.
 */

function updateColors(currentTimeInFrame) {
    var currentActivity = activitiesData[currentTimeInFrame];
    var col = ColSchInfo();
    var activityRange = ColSchGetBounds();
    SHADING_Context.colorscheme_set_uniforms(GL_shaderProgram, activityRange.min, activityRange.max, activityRange.bins);

    if (isOneToOneMapping) {
        for (var i = 0; i < brainBuffers.length; i++) {
            var upperBorder = brainBuffers[i][0].numItems / 3;
            var offset_start = i * 40000;
            var currentActivitySlice = currentActivity.slice(offset_start, offset_start + upperBorder);
            var activity = new Float32Array(currentActivitySlice);

            gl.bindBuffer(gl.ARRAY_BUFFER, brainBuffers[i][3]);
            gl.bufferData(gl.ARRAY_BUFFER, activity, gl.STATIC_DRAW);
            gl.uniform1f(GL_shaderProgram.colorSchemeUniform, col.tex_v);
        }
    } else {
        for (var ii = 0; ii < NO_OF_MEASURE_POINTS; ii++) {
            if(VS_selectedRegions.indexOf(ii) !== -1){
                gl.uniform2f(GL_shaderProgram.activityUniform[ii], currentActivity[ii], col.tex_v);
            }else{
                gl.uniform2f(GL_shaderProgram.activityUniform[ii], currentActivity[ii], col.muted_tex_v);
            }
        }
        // default color for a measure point
        gl.uniform2f(GL_shaderProgram.activityUniform[NO_OF_MEASURE_POINTS], activityMin, col.measurePoints_tex_v);
        // color used for a picked measure point
        gl.uniform2f(GL_shaderProgram.activityUniform[NO_OF_MEASURE_POINTS + 1], activityMax, col.measurePoints_tex_v);
    }
}

function toggleMeasureNodes() {
    displayMeasureNodes = ! displayMeasureNodes;
}


function switchFaceObject() {
    isFaceToDisplay = !isFaceToDisplay;
}

/**
 * Draw model with filled Triangles of isolated Points (Vertices).
 */
function wireFrame() {
    if (drawingMode === gl.POINTS) {
        drawingMode = gl.TRIANGLES;
    } else {
        drawingMode = gl.POINTS;
    }
}

/**
 * Sets a new movie speed.
 * To stop the movie set AG_isStopped to true rather than passing 0 here.
 */
function setTimeStep(newTimeStepsPerTick) {
    timeStepsPerTick = newTimeStepsPerTick;
    if (timeStepsPerTick < 1){ // subunit speed
        TIME_STEP = 1;
    }else{
        TIME_STEP = Math.floor(timeStepsPerTick);
    }
}

function resetSpeedSlider() {
    setTimeStep(1);
    $("#sliderStep").slider("option", "value", 1);
    refreshCurrentDataSlice();
}

function setNavigatorVisibility(enable) {
    drawNavigator = enable;
}

function toggleDrawTriangleLines() {
    drawTriangleLines = !drawTriangleLines;
}

function toggleDrawBoundaries() {
    drawBoundaries = !drawBoundaries;
}

function setSpecularHighLights(enable){
    drawSpeculars = enable;
}

/**
 * Creates a list of webGl buffers.
 *
 * @param dataList a list of lists. Each list will contain the data needed for creating a gl buffer.
 */
function createWebGlBuffers(dataList) {
    var result = [];
    for (var i = 0; i < dataList.length; i++) {
        var buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(dataList[i]), gl.STATIC_DRAW);
        buffer.numItems = dataList[i].length;
        result.push(buffer);
    }

    return result;
}

/**
 * Read data from the specified urls.
 *
 * @param data_url_list a list of urls from where   it should read the data
 * @param staticFiles <code>true</code> if the urls points to some static files
 */
function readFloatData(data_url_list, staticFiles) {
    var result = [];
    for (var i = 0; i < data_url_list.length; i++) {
        var data_json = HLPR_readJSONfromFile(data_url_list[i], staticFiles);
        if (staticFiles) {
            for (var j = 0; j < data_json.length; j++) {
                data_json[j] = parseFloat(data_json[j]);
            }
        }
        result.push(data_json);
        data_json = null;
    }
    return result;
}

/**
 * Computes the data for alpha and alphasIndices.
 *
 * @param vertices a list which contains lists of vertices. E.g.: [[slice_1_vertices],...,[slice_n_vertices]]
 * @param measurePoints a list which contains all the measure points. E.g.: [[x0,y0,z0],[x1,y1,z1],...]
 */
function computeVertexRegionMap(vertices, measurePoints) {
    var vertexRegionMap = [];
    for (var i = 0; i < vertices.length; i++) {
        var reg = [];
        for (var j = 0; j < vertices[i].length/3; j++) {
            var currentVertex = vertices[i].slice(j * 3, (j + 1) * 3);
            var closestPosition = NAV_BrainNavigator.findClosestPosition(currentVertex, measurePoints);
            reg.push(closestPosition, 0, 0);
        }
        vertexRegionMap.push(reg);
    }
    return vertexRegionMap;
}


/**
 * Method used for creating a color buffer for a cube (measure point).
 *
 * @param isPicked If <code>true</code> then the color used will be
 * the one used for drawing the measure points for which the
 * corresponding eeg channels are selected.
 */
function createColorBufferForCube(isPicked) {
    var pointColor = [];
    if (isOneToOneMapping) {
        pointColor = [0.34, 0.95, 0.37, 1.0];
        if (isPicked) {
            pointColor = [0.99, 0.99, 0.0, 1.0];
        }
    } else {
        pointColor = [NO_OF_MEASURE_POINTS, 0, 0];
        if (isPicked) {
            pointColor = [NO_OF_MEASURE_POINTS + 1, 0, 0];
        }
    }
    var colors = [];
    for (var i = 0; i < 24; i++) {
        colors = colors.concat(pointColor);
    }
    var cubeColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cubeColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    return cubeColorBuffer;
}


function bufferAtPoint(p) {
    var result = HLPR_bufferAtPoint(gl, p);
    var bufferVertices= result[0];
    var bufferNormals = result[1];
    var bufferTriangles = result[2];

    if (isOneToOneMapping) {
        return [bufferVertices, bufferNormals, bufferTriangles, createColorBufferForCube(false)];
    } else {
        return [bufferVertices, bufferNormals, bufferTriangles, null, createColorBufferForCube(false)];
    }
}


function initBuffers(urlVertices, urlNormals, urlTriangles, urlRegionMap, staticFiles) {
    var verticesData = readFloatData(urlVertices, staticFiles);
    var vertices = createWebGlBuffers(verticesData);
    var normals = HLPR_getDataBuffers(gl, urlNormals, staticFiles);
    var indexes = HLPR_getDataBuffers(gl, urlTriangles, staticFiles, true);
    
    // Fake buffers, copy of the normals, in case of transparency, we only need dummy ones.
    var vertexRegionMap = normals;
    // warning: these 'fake' buffers will be used and rendered when region colored surfaces are shown.
    // This happens for all static surface viewers. The reason we do not have weird coloring effects
    // is that normals have subunitary components that are truncated to 0 in the shader.
    // todo: accidental use of the fake buffers should be visible. consider uvec3 in shader
    if (!isOneToOneMapping && urlRegionMap && urlRegionMap.length) {
        vertexRegionMap = HLPR_getDataBuffers(gl, urlRegionMap);
    } else if (isEEGView) {
        // if is eeg view than we use the static surface 'eeg_skin_surface' and we have to compute the vertexRegionMap;
        // todo: do this on the server to eliminate this special case
        var regionData = computeVertexRegionMap(verticesData, measurePoints);
        vertexRegionMap = createWebGlBuffers(regionData);
    }
    var result = [];
    for (var i=0; i< vertices.length; i++) {
        if (isOneToOneMapping) {
            var activityBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, activityBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices[i].numItems), gl.STATIC_DRAW);
            result.push([vertices[i], normals[i], indexes[i], activityBuffer]);
        } else {
            result.push([vertices[i], normals[i], indexes[i], null, vertexRegionMap[i]]);
        }
    }
    return result;
}


function initRegionBoundaries(boundariesURL) {
    if (boundariesURL) {
        doAjaxCall({
            url: boundariesURL,
            async: true,
            success: function(data) {
                data = $.parseJSON(data);
                var boundaryVertices = data[0];
                var boundaryEdges = data[1];
                var boundaryNormals = data[2];
                for (var i = 0; i < boundaryVertices.length; i++) {
                    boundaryVertexBuffers.push(HLPR_createWebGlBuffer(gl, boundaryVertices[i], false, false));
                    boundaryNormalsBuffers.push(HLPR_createWebGlBuffer(gl, boundaryNormals[i], false, false));
                    boundaryEdgesBuffers.push(HLPR_createWebGlBuffer(gl, boundaryEdges[i], true, false));
                }
            }
        });
    }
}

/**
 * Make a draw call towards the GL_shaderProgram compiled from common/vertex_shader common_fragment_shader
 * Note: all attributes have to be bound even if the shader does not explicitly use them (ex picking mode)
 * @param drawMode Triangles / Points
 * @param buffers Buffers to be drawn. Array of (vertices, normals, triangles, colors) for one to one mappings
 *                Array of (vertices, normals, triangles, alphas, alphaindices) for region based drawing
 */
function drawBuffer(drawMode, buffers){
    setMatrixUniforms();
    if (isOneToOneMapping) {
        SHADING_Context.one_to_one_program_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[3], buffers[2], drawMode);
    } else {
        SHADING_Context.region_progam_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[4], buffers[2], drawMode);
    }
}

/**
 *
 * @param drawMode Triangles / Points
 * @param buffersSets Actual buffers to be drawn. Array or (vertices, normals, triangles)
 * @param [bufferSetsMask] Optional. If this array has a 0 at index i then the buffer at index i is not drawn
 * @param [useBlending] When true, the object is drawn with blending (for transparency)
 * @param [cullFace] When gl.FRONT, it will mark current object to be drown twice (another with gl.BACK).
 *                 It should be set to GL.FRONT for objects transparent and convex.
 */
function drawBuffers(drawMode, buffersSets, bufferSetsMask, useBlending, cullFace) {
    if (useBlending) {
        var lightSettings = setLighting(blendingLightSettings);
        gl.enable(gl.BLEND);
        gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        // Blending function for alpha: transparent pix blended over opaque -> opaque pix
        if (cullFace) {
            gl.enable(gl.CULL_FACE);
            gl.cullFace(cullFace);
        }
    }

    for (var i = 0; i < buffersSets.length; i++) {
        if(bufferSetsMask != null && !bufferSetsMask[i]){
            continue;
        }
        drawBuffer(drawMode, buffersSets[i]);
    }

    if (useBlending) {
        gl.disable(gl.BLEND);
        gl.disable(gl.CULL_FACE);
        setLighting(lightSettings);
        // Draw the same transparent object the second time
        if (cullFace === gl.FRONT) {
            drawBuffers(drawMode, buffersSets, bufferSetsMask, useBlending, gl.BACK);
        }
    }
}


function drawRegionBoundaries() {
    if (boundaryVertexBuffers && boundaryEdgesBuffers) {
        if (drawingMode !== gl.POINTS) {
            // Usually draw the boundaries with the same color. But in points mode draw them with the vertex colors.
            var lightSettings = setLighting(regionLinesLightSettings);
        }
        gl.lineWidth(3.0);
        // replace the vertex, normal and element buffers from the brain buffer set. Keep the alpha buffers
        var bufferSets = [];
        for (var c = 0; c < brainBuffers.length; c++){
            var chunk = brainBuffers[c].slice();
            chunk[0] = boundaryVertexBuffers[c];
            chunk[1] = boundaryNormalsBuffers[c];
            chunk[2] = boundaryEdgesBuffers[c];
            bufferSets.push(chunk);
        }
        drawBuffers(gl.LINES, bufferSets, bufferSetsMask);
        if (drawingMode !== gl.POINTS) {
            setLighting(lightSettings); // we've drawn solid colors, now restore previous lighting
        }
    } else {
        displayMessage('Boundaries data not yet loaded. Display will refresh automatically when load is finished.', 'infoMessage');
    }
}


function drawBrainLines(linesBuffers, brainBuffers, bufferSetsMask) {
    if (drawingMode !== gl.POINTS) {
        // Usually draw the wireframe with the same color. But in points mode draw with the vertex colors.
        var lightSettings = setLighting(linesLightSettings);
    }
    gl.lineWidth(1.0);
    // we want all the brain buffers in this set except the element array buffer (at index 2)
    var bufferSets = [];
    for (var c = 0; c < brainBuffers.length; c++){
        var chunk = brainBuffers[c].slice();
        chunk[2] = linesBuffers[c];
        bufferSets.push(chunk);
    }
    drawBuffers(gl.LINES, bufferSets, bufferSetsMask);
    if (drawingMode !== gl.POINTS) {
        setLighting(lightSettings);
    }
}

/**
 * Actual scene drawing step.
 */
function tick() {

    if (sliderSel) {
        return;
    }

    //// Update activity buffers to be drawn at next step
    // If we are in the middle of waiting for the next data file just
    // stop and wait since we might have an index that is 'out' of this data slice
    if (! AG_isStopped ) {
        // Synchronizes display time with movie time
        var shouldStep = false;
        if (timeStepsPerTick >= 1) {
            shouldStep = true;
        } else if (elapsedTicksPerTimeStep >= (1 / timeStepsPerTick)) {
            shouldStep = true;
            elapsedTicksPerTimeStep = 0;
        } else {
            elapsedTicksPerTimeStep += 1;
        }

        if (shouldStep && shouldIncrementTime) {
            currentTimeValue = currentTimeValue + TIME_STEP;
        }

        if (currentTimeValue > MAX_TIME) {
            // Next time value is no longer in activity data.
            initActivityData();
            if (isDoubleView) {
                loadEEGChartFromTimeStep(0);
                drawGraph(false, 0);
            }
            shouldStep = false;
        }

        if (shouldStep) {
            if (shouldLoadNextActivitiesFile()) {
                loadNextActivitiesFile();
            }
            if (shouldChangeCurrentActivitiesFile()) {
                changeCurrentActivitiesFile();
            }
            if (isDoubleView) {
                drawGraph(true, TIME_STEP);
            }
        }
    }

    var currentTimeInFrame = Math.floor((currentTimeValue - totalPassedActivitiesData) / TIME_STEP);
    updateColors(currentTimeInFrame);

    drawScene();

    /// Update FPS and Movie timeline
    if (!isPreview) {
        var timeNow = new Date().getTime();
        var elapsed = timeNow - lastTime;

        if (lastTime !== 0) {
            framestime.shift();
            framestime.push(elapsed);
        }

        lastTime = timeNow;
        if (timeData.length > 0 && ! AG_isStopped) {
            document.getElementById("TimeNow").value = toSignificantDigits(timeData[currentTimeValue], 2);
        }
        var meanFrameTime = 0;
        for(var i=0; i < framestime.length; i++){
            meanFrameTime += framestime[i];
        }
        meanFrameTime = meanFrameTime / framestime.length;
        document.getElementById("FramesPerSecond").innerHTML = Math.floor(1000/meanFrameTime).toFixed();
        if (! sliderSel && ! AG_isStopped) {
            $("#slider").slider("option", "value", currentTimeValue);
        }
    }
}

/**
 * Draw from buffers.
 */
function drawScene() {

    var theme = ColSchGetTheme().surfaceViewer;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);

    // Draw sections before setting the correct draw perspective, to work with "rel-time refresh of sections"
    VB_BrainNavigator.maybeRefreshSections();

    // View angle is 45, we want to see object from near up to 800 distance from camera
    perspective(45, gl.viewportWidth / gl.viewportHeight, near, 800.0);

    mvPushMatrix();
    mvRotate(180, [0, 0, 1]);

    if (!doPick) {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        if (drawSpeculars){
            setLighting(specularLightSettings);
        }else{
            setLighting();
        }

        if(VS_showLegend){
            mvPushMatrix();
            loadIdentity();
            drawBuffers(gl.TRIANGLES, [LEG_legendBuffers]);
            mvPopMatrix();
        }

        if(isInternalSensorView){
            // for internal sensors we render only the sensors
            drawBuffers(gl.TRIANGLES, measurePointsBuffers);
        } else {
            // draw surface
            drawBuffers(drawingMode, brainBuffers, bufferSetsMask);

            if (drawBoundaries) {
                drawRegionBoundaries();
            }
            if (drawTriangleLines) {
                drawBrainLines(brainLinesBuffers, brainBuffers, bufferSetsMask);
            }
            if (displayMeasureNodes) {
                drawBuffers(gl.TRIANGLES, measurePointsBuffers);
            }
        }

        if (isFaceToDisplay) {
            var faceDrawMode = isInternalSensorView ? drawingMode : gl.TRIANGLES;
            mvPushMatrix();
            mvTranslate(VB_BrainNavigator.getPosition());
            drawBuffers(faceDrawMode, shelfBuffers, null, true, gl.FRONT);
            mvPopMatrix();
        }

        if (drawNavigator) {
            VB_BrainNavigator.drawNavigator();
        }

    } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, GL_colorPickerBuffer);
        gl.disable(gl.BLEND);
        gl.disable(gl.DITHER);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        setLighting(pickingLightSettings);

        if (GL_colorPickerInitColors.length === 0) {
            GL_initColorPickingData(NO_OF_MEASURE_POINTS);
        }    

        for (var i = 0; i < NO_OF_MEASURE_POINTS; i++){
            var mpColor = GL_colorPickerInitColors[i];
            gl.uniform4fv(GL_shaderProgram.materialColor, mpColor);
            drawBuffer(gl.TRIANGLES, measurePointsBuffers[i]);
        }
        VS_pickedIndex = GL_getPickedIndex();
        doPick = false;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    mvPopMatrix();
}

////////////////////////////////////////~~~~~~~~~ END WEB GL RELATED RENDERING ~~~~~~~/////////////////////////////////


/////////////////////////////////////// ~~~~~~~~~~ DATA RELATED METHOD ~~~~~~~~~~~~~ //////////////////////////////////

/**
 * Change the currently selected state variable. Get the newly selected value, reset the currentTimeValue to start
 * and read the first page of the new mode/state var combination.
 */
function VS_changeStateVariable(id, val) {
    selectedStateVar = val;
    $("#slider").slider("option", "value", currentTimeValue);
    initActivityData();
}

/**
 * Change the currently selected mode. Get the newly selected value, reset the currentTimeValue to start
 * and read the first page of the new mode/state var combination.
 */
function VS_changeMode(id, val) {
    selectedMode = val;
    $("#slider").slider("option", "value", currentTimeValue);
    initActivityData();
}

/**
 * Just read the first slice of activity data and set the time step to 0.
 */
function initActivityData() {
    currentTimeValue = 0;
    //read the first file
    var initUrl = getUrlForPageFromIndex(0);
    activitiesData = HLPR_readJSONfromFile(initUrl);
    if (activitiesData != null) {
        currentActivitiesFileLength = activitiesData.length * TIME_STEP;
        totalPassedActivitiesData = 0;
    }
}

/**
 * Load the brainviewer from this given time step.
 */
function loadFromTimeStep(step) {
    showBlockerOverlay(50000);
    if (step % TIME_STEP !== 0) {
        step = step - step % TIME_STEP + TIME_STEP; // Set time to be multiple of step
    }
    var nextUrl = getUrlForPageFromIndex(step);
    currentAsyncCall = null;
    readFileData(nextUrl, false);
    currentTimeValue = step;
    activitiesData = nextActivitiesFileData.slice(0);
    nextActivitiesFileData = null;
    currentActivitiesFileLength = activitiesData.length * TIME_STEP;
    totalPassedActivitiesData = currentTimeValue;
    // Also sync eeg monitor if in double view
    if (isDoubleView) {
        loadEEGChartFromTimeStep(step);
    }
    closeBlockerOverlay();
}

/**
 * Refresh the current data with the new time step.
 */
function refreshCurrentDataSlice() {
    if (currentTimeValue % TIME_STEP !== 0) {
        currentTimeValue = currentTimeValue - currentTimeValue % TIME_STEP + TIME_STEP; // Set time to be multiple of step
    }
    loadFromTimeStep(currentTimeValue);
}

/**
 * Generate the url that reads one page of data starting from @param index
 */
function getUrlForPageFromIndex(index) {
    var fromIdx = index;
    if (fromIdx > MAX_TIME) {
        fromIdx = 0;
    }
    var toIdx = fromIdx + pageSize * TIME_STEP;
    return readDataPageURL(urlBase, fromIdx, toIdx, selectedStateVar, selectedMode, TIME_STEP);
}

/**
 * If we are at the last NEXT_PAGE_THRESHOLD points of data we should start loading the next data file
 * to get an animation as smooth as possible.
 */
function shouldLoadNextActivitiesFile() {

    if (!isPreview && (currentAsyncCall == null) && ((currentTimeValue - totalPassedActivitiesData + NEXT_PAGE_THREASHOLD * TIME_STEP) >= currentActivitiesFileLength)) {
        if (nextActivitiesFileData == null || nextActivitiesFileData.length === 0) {
            return true;
        }
    }
    return false;
}

/**
 * Start a new async call that should load required data for the next activity slice.
 */
function loadNextActivitiesFile() {
    var nextFileIndex = totalPassedActivitiesData + currentActivitiesFileLength;
    var nextUrl = getUrlForPageFromIndex(nextFileIndex);
    var asyncCallId = new Date().getTime();
    currentAsyncCall = asyncCallId;
    readFileData(nextUrl, true, asyncCallId);
}

/**
 * If the next time value is bigger that the length of the current activity loaded data
 * that means it's time to switch to the next activity data slice.
 */
function shouldChangeCurrentActivitiesFile() {
    return ((currentTimeValue + TIME_STEP - totalPassedActivitiesData) >= currentActivitiesFileLength);
}

/**
 * We've reached the end of the current activity chunk. Time to switch to
 * the next one.
 */
function changeCurrentActivitiesFile() {
    if (nextActivitiesFileData == null || !nextActivitiesFileData.length ) {
        // Async data call was not finished, stop incrementing call and wait for data.
        shouldIncrementTime = false;
        return;
    }

    activitiesData = nextActivitiesFileData.slice(0);
    nextActivitiesFileData = null;
    totalPassedActivitiesData = totalPassedActivitiesData + currentActivitiesFileLength;
    currentActivitiesFileLength = activitiesData.length * TIME_STEP;
    currentAsyncCall = null;
    if (activitiesData && activitiesData.length ) {
        shouldIncrementTime = true;
    }
    if (totalPassedActivitiesData >= MAX_TIME) {
        totalPassedActivitiesData = 0;
    }
}


function readFileData(fileUrl, async, callIdentifier) {
    nextActivitiesFileData = null;
    // Keep a call identifier so we don't "intersect" async calls when two
    // async calls are started before the first one finishes.
    var self = this;
    self.callIdentifier = callIdentifier;
    doAjaxCall({
        url: fileUrl,
        async: async,
        success: function(data) {
            if ((self.callIdentifier === currentAsyncCall) || !async) {
                nextActivitiesFileData = eval(data);
                data = null;
            }
        }
    });
}


/////////////////////////////////////// ~~~~~~~~~~ END DATA RELATED METHOD ~~~~~~~~~~~~~ //////////////////////////////////
