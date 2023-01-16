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

/* The comment below lists the global functions used in this file.
 * It is here to make jshint happy and to document these implicit global dependencies.
 * In the future we might group these into namespace objects.
 * ( Global state is not in this list except gl and namespaces; let them be warnings )
 */

/* globals gl, SHADING_Context, GL_shaderProgram, displayMessage, HLPR_readJSONfromFile, readDataPageURL,
 GL_handleKeyDown, GL_handleKeyUp, GL_handleMouseMove, GL_handleMouseWeel,
 initGL, updateGLCanvasSize, LEG_updateLegendVerticesBuffers,
 basicInitShaders, basicInitSurfaceLighting, GL_initColorPickFrameBuffer,
 ColSchGetTheme, LEG_generateLegendBuffers, LEG_initMinMax
 */

/**
 * WebGL methods "inheriting" from webGL_xx.js in static/js.
 */
var BRAIN_CANVAS_ID = "GLcanvas";
/**
 * Variables for displaying Time and computing Frames/Sec
 */
var lastTime = 0;
var framestime = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50];

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
/**
 * This buffer arrays will contain:
 * arr[i][0] Vertices buffer
 * arr[i][1] Normals buffer
 * arr[i][2] Triangles indices buffer
 * arr[i][3] Color buffer (same length as vertices /3 * 4) in case of one-to-one mapping
 * arr[i][3] Region indexes, when not one-to-one mapping
 */
var brainBuffers = [];
var brainLinesBuffers = [];
var shellBuffers = [];
var measurePointsBuffers = [];

var regionBoundariesController = null;

var activitiesData = [], timeData = [], measurePoints = [], measurePointsLabels = [];

var pageSize = 0;
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

var urlBaseAdapter = '';


function VS_init_hemisphere_mask(hemisphere_chunk_mask) {
    VS_hemisphere_chunk_mask = hemisphere_chunk_mask;
    if (hemisphere_chunk_mask !== null && hemisphere_chunk_mask !== undefined) {
        bufferSetsMask = [];
        for (let i = 0; i < VS_hemisphere_chunk_mask.length; i++) {
            bufferSetsMask[i] = 1;
        }
    }
}

function VS_SetHemisphere(h) {
    VS_hemisphereVisibility = h;
    for (let i = 0; i < VS_hemisphere_chunk_mask.length; i++) {
        if (h === null || h === undefined) {
            bufferSetsMask[i] = 1;
        } else if (h === 'l') {
            bufferSetsMask[i] = 1 - VS_hemisphere_chunk_mask[i];
        } else if (h === 'r') {
            bufferSetsMask[i] = VS_hemisphere_chunk_mask[i];
        }
    }
}

function _VS_static_entrypoint(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                               noOfMeasurePoints, urlRegionMapList, urlMeasurePointsLabels, boundaryURL, shellObject,
                               hemisphereChunkMask, argDisplayMeasureNodes, argIsFaceToDisplay,
                               minMeasure, maxMeasure, urlMeasure) {
    // initialize global configuration
    isDoubleView = false;
    isOneToOneMapping = false;
    shouldIncrementTime = false;
    AG_isStopped = true;
    displayMeasureNodes = argDisplayMeasureNodes;
    isFaceToDisplay = argIsFaceToDisplay; // this could be retrieved from the dom like drawNavigator
    // make checkbox consistent with this flag
    $("#displayFaceChkId").attr('checked', isFaceToDisplay);
    drawNavigator = $("#showNavigator").prop('checked');

    if (noOfMeasurePoints === 0) {
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
        let measure;
        if (urlMeasure === '') {
            // Empty url => The static viewer has to show a region map.
            // The measure will be a range(NO_OF_MEASURE_POINTS)
            measure = [];
            for (let i = 0; i < NO_OF_MEASURE_POINTS; i++) {
                measure.push(i);
            }
        } else {
            measure = HLPR_readJSONfromFile(urlMeasure);
        }
        // The activity data will contain just one frame containing the values of the connectivity measure.
        activitiesData = [measure];
    }

    VS_showLegend = false;
    if (parseFloat(minMeasure) < parseFloat(maxMeasure)) {
        const brainLegendDiv = document.getElementById('brainLegendDiv');
        ColSch_updateLegendLabels(brainLegendDiv, minMeasure, maxMeasure, "100%");
        VS_showLegend = true;
    }

    for (let i = 0; i < NO_OF_MEASURE_POINTS; i++) {
        VS_selectedRegions.push(i);
    }

    const canvas = document.getElementById(BRAIN_CANVAS_ID);
    _initViewerGL(canvas, urlVerticesList, urlNormalsList, urlTrianglesList,
        urlRegionMapList, urlLinesList, boundaryURL, shellObject, hemisphereChunkMask);

    _bindEvents(canvas);

    //specify the re-draw function.
    if (_isValidActivityData()) {
        setInterval(tick, TICK_STEP);
    }
}

function _VS_movie_entrypoint(baseAdapterURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                              urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                              urlRegionMapList, minActivity, maxActivity, oneToOneMapping, doubleView,
                              shellObject, hemisphereChunkMask, urlMeasurePointsLabels, boundaryURL) {
    // initialize global configuration
    isDoubleView = doubleView;
    if (oneToOneMapping === 'True') {
        isOneToOneMapping = true;
    }
    // these global flags could be structured better
    isEEGView = isDoubleView && !isInternalSensorView;
    activityMin = parseFloat(minActivity);
    activityMax = parseFloat(maxActivity);
    pageSize = onePageSize;
    urlBaseAdapter = baseAdapterURL;

    // initialize global data
    _initMeasurePoints(noOfMeasurePoints, urlMeasurePoints, urlMeasurePointsLabels);
    _initTimeData(urlTimeList);
    initActivityData();

    if (isDoubleView) {
        $("#displayFaceChkId").trigger('click');
    }
    drawNavigator = $("#showNavigator").prop('checked');

    const canvas = document.getElementById(BRAIN_CANVAS_ID);

    _initViewerGL(canvas, urlVerticesList, urlNormalsList, urlTrianglesList,
        urlRegionMapList, urlLinesList, boundaryURL, shellObject, hemisphereChunkMask);

    _bindEvents(canvas);

    _initSliders();

    //specify the re-draw function.
    if (_isValidActivityData()) {
        setInterval(tick, TICK_STEP);
    }
}

function _VS_init_cubicalMeasurePoints() {
    for (let i = 0; i < NO_OF_MEASURE_POINTS; i++) {
        const result = HLPR_bufferAtPoint(gl, measurePoints[i]);
        const bufferVertices = result[0];
        const bufferNormals = result[1];
        const bufferTriangles = result[2];
        const bufferColor = createColorBufferForCube(false);
        measurePointsBuffers[i] = [bufferVertices, bufferNormals, bufferTriangles, bufferColor];
    }
}

function VS_StartSurfaceViewer(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                               noOfMeasurePoints, urlRegionMapList, urlMeasurePointsLabels,
                               boundaryURL, shelveObject, minMeasure, maxMeasure, urlMeasure, hemisphereChunkMask) {

    _VS_static_entrypoint(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
        noOfMeasurePoints, urlRegionMapList, urlMeasurePointsLabels, boundaryURL, shelveObject,
        hemisphereChunkMask, false, false, minMeasure, maxMeasure, urlMeasure);
    _VS_init_cubicalMeasurePoints();
}

function VS_StartEEGSensorViewer(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
                                 noOfMeasurePoints, urlMeasurePointsLabels,
                                 shellObject, minMeasure, maxMeasure, urlMeasure) {
    isEEGView = true;
    _VS_static_entrypoint(urlVerticesList, urlLinesList, urlTrianglesList, urlNormalsList, urlMeasurePoints,
        noOfMeasurePoints, '', urlMeasurePointsLabels, '', shellObject, null, true, true,
        minMeasure, maxMeasure, urlMeasure);
    _VS_init_cubicalMeasurePoints();
}

function VS_StartBrainActivityViewer(baseAdapterURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
                                     urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
                                     urlRegionMapList, minActivity, maxActivity,
                                     oneToOneMapping, doubleView, shellObject, hemisphereChunkMask,
                                     urlMeasurePointsLabels, boundaryURL, measurePointsSelectionGID) {
    _VS_movie_entrypoint(baseAdapterURL, onePageSize, urlTimeList, urlVerticesList, urlLinesList,
        urlTrianglesList, urlNormalsList, urlMeasurePoints, noOfMeasurePoints,
        urlRegionMapList, minActivity, maxActivity,
        oneToOneMapping, doubleView, shellObject, hemisphereChunkMask,
        urlMeasurePointsLabels, boundaryURL);
    _VS_init_cubicalMeasurePoints();

    if (!isDoubleView) {
        // If this is a brain activity viewer then we have to initialize the selection component
        _initChannelSelection(measurePointsSelectionGID);
        // For the double view the selection is the responsibility of the extended view functions
    }
}

function _isValidActivityData() {
    if (isOneToOneMapping) {
        if (activitiesData.length !== brainBuffers.length) {
            displayMessage("The number of activity buffers should equal the number of split surface slices", "errorMessage");
            return false;
        }
        if (3 * activitiesData[0][0].length !== brainBuffers[0][0].numItems) {
            displayMessage("The number of activity points should equal the number of surface vertices", "errorMessage");
            return false;
        }
    } else {
        if (NO_OF_MEASURE_POINTS !== activitiesData[0].length) {
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
                       urlRegionMapList, urlLinesList, boundaryURL, shellObject, hemisphere_chunk_mask) {
    customInitGL(canvas);
    GL_initColorPickFrameBuffer();
    initShaders();

    if (VS_showLegend) {
        LEG_initMinMax(activityMin, activityMax);
        ColSch_initColorSchemeGUI(activityMin, activityMax, LEG_updateLegendColors);
        LEG_generateLegendBuffers();
    } else {
        ColSch_initColorSchemeGUI(activityMin, activityMax);
    }

    if (urlVerticesList) {
        let parsedIndices = [];
        if (urlRegionMapList) {
            parsedIndices = $.parseJSON(urlRegionMapList);
        }
        brainBuffers = initBuffers($.parseJSON(urlVerticesList), $.parseJSON(urlNormalsList),
            $.parseJSON(urlTrianglesList), parsedIndices, isDoubleView);
    }

    VS_init_hemisphere_mask(hemisphere_chunk_mask);

    brainLinesBuffers = HLPR_getDataBuffers(gl, $.parseJSON(urlLinesList), isDoubleView, true);
    regionBoundariesController = new RB_RegionBoundariesController(boundaryURL);

    if (shellObject) {
        shellObject = $.parseJSON(shellObject);
        shellBuffers = initBuffers(shellObject[0], shellObject[1], shellObject[2], false, true);
    }

    VB_BrainNavigator = new NAV_BrainNavigator(isOneToOneMapping, brainBuffers, measurePoints, measurePointsLabels);
}


function _bindEvents(canvas) {
    // Enable keyboard and mouse interaction
    canvas.onkeydown = GL_handleKeyDown;
    canvas.onkeyup = GL_handleKeyUp;
    canvas.onmousedown = customMouseDown;
    $(document).on('mouseup', customMouseUp);
    $(canvas).on('contextmenu', _onContextMenu);
    $(document).on('mousemove', GL_handleMouseMove);

    $(canvas).mousewheel(function (event, delta) {
        GL_handleMouseWeel(delta);
        return false; // prevent default
    });

    if (!isDoubleView) {
        const canvasX = document.getElementById('brain-x');
        if (canvasX) {
            canvasX.onmousedown = function (event) {
                VB_BrainNavigator.moveInXSection(event)
            };
        }
        const canvasY = document.getElementById('brain-y');
        if (canvasY) {
            canvasY.onmousedown = function (event) {
                VB_BrainNavigator.moveInYSection(event)
            };
        }
        const canvasZ = document.getElementById('brain-z');
        if (canvasZ) {
            canvasZ.onmousedown = function (event) {
                VB_BrainNavigator.moveInZSection(event)
            };
        }
    }
}

function _initMeasurePoints(noOfMeasurePoints, urlMeasurePoints, urlMeasurePointsLabels) {
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

function _initTimeData(urlTimeList) {
    const timeUrls = $.parseJSON(urlTimeList);
    for (let i = 0; i < timeUrls.length; i++) {
        timeData = timeData.concat(HLPR_readJSONfromFile(timeUrls[i]));
    }
    MAX_TIME = timeData.length - 1;
}

function _updateSpeedSliderValue(stepsPerTick) {
    let s;
    if (stepsPerTick >= 1) {
        s = stepsPerTick.toFixed(0);
    } else {
        s = "1/" + (1 / stepsPerTick).toFixed(0);
    }
    $("#slider-value").html(s);
}

function _initSliders() {
    const maxAllowedTimeStep = Math.ceil(MAX_TIME / ACTIVITY_FRAMES_IN_TIME_LINE_AT_MAX_SPEED);
    // after being converted to the exponential range maxSpeed must not exceed maxAllowedTimeStep
    const maxSpeedSlider = Math.min(10, 5 + Math.log(maxAllowedTimeStep) / Math.LN2);

    if (timeData.length > 0) {
        $("#sliderStep").slider({
            min: 0, max: maxSpeedSlider, step: 1, value: 5,
            stop: function () {
                refreshCurrentDataSlice();
                sliderSel = false;
            },
            slide: function (event, target) {
                // convert the linear 0..10 range to the exponential 1/32..1..32 range
                const newStep = Math.pow(2, target.value - 5);
                setTimeStep(newStep);
                _updateSpeedSliderValue(timeStepsPerTick);
                sliderSel = true;
            }
        });
        // Initialize slider for timeLine
        $("#slider").slider({
            min: 0, max: MAX_TIME,
            slide: function (event, target) {
                sliderSel = true;
                currentTimeValue = target.value;
                $('#TimeNow').val(currentTimeValue);
            },
            stop: function (event, target) {
                sliderSel = false;
                loadFromTimeStep(target.value);
            }
        });
    } else {
        $("#divForSliderSpeed").hide();
    }
    _updateSpeedSliderValue(timeStepsPerTick);

    $('#TimeNow').click(function () {
        if (!AG_isStopped) {
            pauseMovie();
        }
        $(this).select();
    }).change(function (ev) {
        let val = parseFloat(ev.target.value);
        if (val === null || val < 0 || val > MAX_TIME) {
            val = 0;
            ev.target.value = 0;
        }
        $('#slider').slider('value', val);
        loadFromTimeStep(val);
    });
}

function _initChannelSelection(selectionGID) {
    const vs_regionsSelector = TVBUI.regionSelector("#channelSelector", {filterGid: selectionGID});

    vs_regionsSelector.change(function (value) {
        VS_selectedRegions = [];
        for (let i = 0; i < value.length; i++) {
            VS_selectedRegions.push(parseInt(value[i], 10));
        }
    });
    //sync region filter with initial selection
    VS_selectedRegions = [];
    const selection = vs_regionsSelector.val();
    for (let i = 0; i < selection.length; i++) {
        VS_selectedRegions.push(parseInt(selection[i], 10));
    }
    const mode_selector = TVBUI.modeAndStateSelector("#channelSelector", 0);
    mode_selector.modeChanged(VS_changeMode);
    mode_selector.stateVariableChanged(VS_changeStateVariable);
}

////////////////////////////////////////// GL Initializations //////////////////////////////////////////

function customInitGL(canvas) {
    window.onresize = function () {
        updateGLCanvasSize(BRAIN_CANVAS_ID);
        LEG_updateLegendVerticesBuffers();
    };
    initGL(canvas);
    drawingMode = gl.TRIANGLES;
    gl.newCanvasWidth = canvas.clientWidth;
    gl.newCanvasHeight = canvas.clientHeight;
    canvas.redrawFunctionRef = drawScene;            // interface-like function used in HiRes image exporting
    canvas.multipleImageExport = VS_multipleImageExport;

    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
}

/** This callback handles image exporting from this canvas.*/
function VS_multipleImageExport(saveFigure) {
    const canvas = this;

    function saveFrontBack(nameFront, nameBack) {
        mvPushMatrix();
        // front
        canvas.drawForImageExport();
        saveFigure({suggestedName: nameFront});
        // back: rotate model around the vertical y axis in trackball space (almost camera space: camera has a z translation)
        const r = createRotationMatrix(180, [0, 1, 0]);
        GL_mvMatrix = GL_cameraMatrix.x(r.x(GL_trackBallMatrix));
        canvas.drawForImageExport();
        saveFigure({suggestedName: nameBack});
        mvPopMatrix();
    }

    // using drawForImageExport because it handles resizing canvas for export
    // It is set on canvas in initGL and defers to drawscene.

    if (VS_hemisphere_chunk_mask !== null) {    // we have 2 hemispheres
        if (VS_hemisphereVisibility === null) {  // both are visible => take them apart when taking picture
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
    } else {
        SHADING_Context.region_program_init(GL_shaderProgram, NO_OF_MEASURE_POINTS, legendGranularity);
    }
}

///////////////////////////////////////~~~~~~~~START MOUSE RELATED CODE~~~~~~~~~~~//////////////////////////////////


function _onContextMenu() {
    if (!displayMeasureNodes || VS_pickedIndex === -1) {
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
    const col = ColSchInfo();
    const activityRange = ColSchGetBounds();
    SHADING_Context.colorscheme_set_uniforms(GL_shaderProgram, activityRange.min, activityRange.max,
        activityRange.bins, activityRange.centralHoleDiameter);

    if (isOneToOneMapping) {
        for (let i = 0; i < brainBuffers.length; i++) {
            const activity = new Float32Array(activitiesData[i][currentTimeInFrame]);
            gl.bindBuffer(gl.ARRAY_BUFFER, brainBuffers[i][3]);
            gl.bufferData(gl.ARRAY_BUFFER, activity, gl.STATIC_DRAW);
            gl.uniform1f(GL_shaderProgram.colorSchemeUniform, col.tex_v);
        }
    } else {
        const currentActivity = activitiesData[currentTimeInFrame];
        for (let ii = 0; ii < NO_OF_MEASURE_POINTS; ii++) {
            if (VS_selectedRegions.indexOf(ii) !== -1) {
                gl.uniform2f(GL_shaderProgram.activityUniform[ii], currentActivity[ii], col.tex_v);
            } else {
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
    displayMeasureNodes = !displayMeasureNodes;
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
    if (timeStepsPerTick < 1) { // subunit speed
        TIME_STEP = 1;
    } else {
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
    regionBoundariesController.toggleBoundariesVisibility();
}

function setSpecularHighLights(enable) {
    drawSpeculars = enable;
}

/**
 * Creates a list of webGl buffers.
 *
 * @param dataList a list of lists. Each list will contain the data needed for creating a gl buffer.
 */
function createWebGlBuffers(dataList) {
    const result = [];
    for (let i = 0; i < dataList.length; i++) {
        const buffer = gl.createBuffer();
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
    const result = [];
    for (let i = 0; i < data_url_list.length; i++) {
        let data_json = HLPR_readJSONfromFile(data_url_list[i], staticFiles);
        if (staticFiles) {
            for (let j = 0; j < data_json.length; j++) {
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
    const vertexRegionMap = [];
    for (let i = 0; i < vertices.length; i++) {
        const reg = [];
        for (let j = 0; j < vertices[i].length / 3; j++) {
            const currentVertex = vertices[i].slice(j * 3, (j + 1) * 3);
            const closestPosition = NAV_BrainNavigator.findClosestPosition(currentVertex, measurePoints);
            reg.push(closestPosition);
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
    let pointColor = [];
    if (isOneToOneMapping) {
        pointColor = [0.34, 0.95, 0.37, 1.0];
        if (isPicked) {
            pointColor = [0.99, 0.99, 0.0, 1.0];
        }
    } else {
        pointColor = [NO_OF_MEASURE_POINTS];
        if (isPicked) {
            pointColor = [NO_OF_MEASURE_POINTS + 1];
        }
    }
    let colors = [];
    for (let i = 0; i < 24; i++) {
        colors = colors.concat(pointColor);
    }
    const cubeColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cubeColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    return cubeColorBuffer;
}

function initBuffers(urlVertices, urlNormals, urlTriangles, urlRegionMap, staticFiles) {
    const verticesData = readFloatData(urlVertices, staticFiles);
    const vertexBatches = createWebGlBuffers(verticesData);
    const normals = HLPR_getDataBuffers(gl, urlNormals, staticFiles);
    const indexes = HLPR_getDataBuffers(gl, urlTriangles, staticFiles, true);

    let vertexRegionMap;
    if (!isOneToOneMapping) {
        if (urlRegionMap && urlRegionMap.length) {
            vertexRegionMap = HLPR_getDataBuffers(gl, urlRegionMap);
        } else if (isEEGView) {
            // if is eeg view than we use the static surface 'eeg_skin_surface' and we have to compute the vertexRegionMap;
            // todo: do this on the server to eliminate this special case
            const regionData = computeVertexRegionMap(verticesData, measurePoints);
            vertexRegionMap = createWebGlBuffers(regionData);
        } else {
            // Fake buffers, copy of the normals, in case of transparency, we only need dummy ones.
            vertexRegionMap = normals;
        }
    }

    const result = [];
    for (let i = 0; i < vertexBatches.length; i++) {
        if (isOneToOneMapping) {
            const activityBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, activityBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexBatches[i].numItems), gl.STATIC_DRAW);
            result.push([vertexBatches[i], normals[i], indexes[i], activityBuffer]);
        } else {
            result.push([vertexBatches[i], normals[i], indexes[i], vertexRegionMap[i]]);
        }
    }
    return result;
}

/**
 * Make a draw call towards the GL_shaderProgram compiled from common/vertex_shader common_fragment_shader
 * Note: all attributes have to be bound even if the shader does not explicitly use them (ex picking mode)
 * @param drawMode Triangles / Points
 * @param buffers Buffers to be drawn. Array of (vertices, normals, triangles, colors) for one to one mappings
 *                Array of (vertices, normals, triangles, alphas, alphaindices) for region based drawing
 */
function drawBuffer(drawMode, buffers) {
    setMatrixUniforms();
    if (isOneToOneMapping) {
        SHADING_Context.one_to_one_program_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[3], buffers[2], drawMode);
    } else {
        SHADING_Context.region_program_draw(GL_shaderProgram, buffers[0], buffers[1], buffers[3], buffers[2], drawMode);
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
    let lightSettings = null;
    if (useBlending) {
        lightSettings = setLighting(blendingLightSettings);
        gl.enable(gl.BLEND);
        gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        // Blending function for alpha: transparent pix blended over opaque -> opaque pix
        if (cullFace) {
            gl.enable(gl.CULL_FACE);
            gl.cullFace(cullFace);
        }
    }

    for (let i = 0; i < buffersSets.length; i++) {
        if (bufferSetsMask !== null && bufferSetsMask !== undefined && !bufferSetsMask[i]) {
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


function drawBrainLines(linesBuffers, brainBuffers, bufferSetsMask) {
    let lightSettings = null;
    if (drawingMode !== gl.POINTS) {
        // Usually draw the wire-frame with the same color. But in points mode draw with the vertex colors.
        lightSettings = setLighting(linesLightSettings);
    }
    gl.lineWidth(1.0);
    // we want all the brain buffers in this set except the element array buffer (at index 2)
    let bufferSets = [];
    for (let c = 0; c < brainBuffers.length; c++) {
        let chunk = brainBuffers[c].slice();
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
    if (!AG_isStopped) {
        // Synchronizes display time with movie time
        let shouldStep = false;
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

    const currentTimeInFrame = Math.floor((currentTimeValue - totalPassedActivitiesData) / TIME_STEP);
    updateColors(currentTimeInFrame);

    drawScene();

    /// Update FPS and Movie timeline
    if (!isPreview) {
        const timeNow = new Date().getTime();
        const elapsed = timeNow - lastTime;

        if (lastTime !== 0) {
            framestime.shift();
            framestime.push(elapsed);
        }

        lastTime = timeNow;
        if (timeData.length > 0 && !AG_isStopped) {
            document.getElementById("TimeNow").value = toSignificantDigits(timeData[currentTimeValue], 2);
        }
        let meanFrameTime = 0;
        for (let i = 0; i < framestime.length; i++) {
            meanFrameTime += framestime[i];
        }
        meanFrameTime = meanFrameTime / framestime.length;
        document.getElementById("FramesPerSecond").innerHTML = Math.floor(1000 / meanFrameTime).toFixed();
        if (!sliderSel && !AG_isStopped) {
            $("#slider").slider("option", "value", currentTimeValue);
        }
    }
}

/**
 * Draw from buffers.
 */
function drawScene() {

    const theme = ColSchGetTheme().surfaceViewer;
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

        if (drawSpeculars) {
            setLighting(specularLightSettings);
        } else {
            setLighting();
        }

        if (VS_showLegend) {
            mvPushMatrix();
            loadIdentity();
            drawBuffers(gl.TRIANGLES, [LEG_legendBuffers]);
            mvPopMatrix();
        }

        if (isInternalSensorView) {
            // for internal sensors we render only the sensors
            drawBuffers(gl.TRIANGLES, measurePointsBuffers);
        } else {
            // draw surface
            drawBuffers(drawingMode, brainBuffers, bufferSetsMask);

            regionBoundariesController.drawRegionBoundaries(drawingMode, brainBuffers);

            if (drawTriangleLines) {
                drawBrainLines(brainLinesBuffers, brainBuffers, bufferSetsMask);
            }
            if (displayMeasureNodes) {
                drawBuffers(gl.TRIANGLES, measurePointsBuffers);
            }
        }

        if (isFaceToDisplay) {
            const faceDrawMode = isInternalSensorView ? drawingMode : gl.TRIANGLES;
            mvPushMatrix();
            mvTranslate(VB_BrainNavigator.getPosition());
            drawBuffers(faceDrawMode, shellBuffers, null, true, gl.FRONT);
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

        for (let i = 0; i < NO_OF_MEASURE_POINTS; i++) {
            const mpColor = GL_colorPickerInitColors[i];
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
    const initUrl = getUrlForPageFromIndex(0);
    activitiesData = HLPR_readJSONfromFile(initUrl);
    if (activitiesData !== null && activitiesData !== undefined) {
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
    const nextUrl = getUrlForPageFromIndex(step);
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
    let fromIdx = index;
    if (fromIdx > MAX_TIME) {
        fromIdx = 0;
    }
    const toIdx = fromIdx + pageSize * TIME_STEP;
    return readDataSplitPageURL(urlBaseAdapter, fromIdx, toIdx, selectedStateVar, selectedMode, TIME_STEP);
}

/**
 * If we are at the last NEXT_PAGE_THRESHOLD points of data we should start loading the next data file
 * to get an animation as smooth as possible.
 */
function shouldLoadNextActivitiesFile() {

    if (!isPreview && (currentAsyncCall === null) && ((currentTimeValue - totalPassedActivitiesData + NEXT_PAGE_THREASHOLD * TIME_STEP) >= currentActivitiesFileLength)) {
        if (nextActivitiesFileData === null || nextActivitiesFileData.length === 0) {
            return true;
        }
    }
    return false;
}

/**
 * Start a new async call that should load required data for the next activity slice.
 */
function loadNextActivitiesFile() {
    const nextFileIndex = totalPassedActivitiesData + currentActivitiesFileLength;
    const nextUrl = getUrlForPageFromIndex(nextFileIndex);
    const asyncCallId = new Date().getTime();
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
    if (nextActivitiesFileData === null || !nextActivitiesFileData.length) {
        // Async data call was not finished, stop incrementing call and wait for data.
        shouldIncrementTime = false;
        return;
    }

    activitiesData = nextActivitiesFileData.slice(0);
    nextActivitiesFileData = null;
    totalPassedActivitiesData = totalPassedActivitiesData + currentActivitiesFileLength;
    currentActivitiesFileLength = activitiesData.length * TIME_STEP;
    currentAsyncCall = null;
    if (activitiesData && activitiesData.length) {
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
    const self = this;
    self.callIdentifier = callIdentifier;
    doAjaxCall({
        url: fileUrl,
        async: async,
        success: function (data) {
            if ((self.callIdentifier === currentAsyncCall) || !async) {
                nextActivitiesFileData = eval(data);
                data = null;
            }
        }
    });
}


/////////////////////////////////////// ~~~~~~~~~~ END DATA RELATED METHOD ~~~~~~~~~~~~~ //////////////////////////////////
