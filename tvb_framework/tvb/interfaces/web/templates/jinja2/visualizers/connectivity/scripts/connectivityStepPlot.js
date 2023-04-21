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

var CONNECTIVITY_SPACE_TIME_CANVAS_ID = "GLcanvas_SPACETIME";

var fullMatrixColors = null;        // Colors for the full connectivity matrix
var outlineVerticeBuffer = null;    // Vertices for the full matrix outline square
var outlineNormalsBuffer = null;    // Normals for the full matrix outline square
var outlineLinesBuffer = null;        // The line indices for thr full matrix outline square
var plotColorBuffers = [];            // A list of color buffers for the various space-time connectivity matrices
var verticesBuffer = [];            // A list of vertex buffers for the various space-time connectivity matrices
var normalsBuffer = [];                // A list of normal buffers for the various space-time connecitivty matrices
var indexBuffer = [];                // A list of triangles index buffers for the various space-time connectivity matrices
var linesIndexBuffer = [];            // A list of line index buffers for the various space-time connectivity matrices

var plotSize = 250;                    // Default plot size of 250 pixels
var defaultZ = -2.0;                // Default plot Z position of -2.0
var doPick = false;                    // No picking by default
var nrOfSteps = 6;                    // Number of space-time plots we will draw in scene
var colorsForPicking = [];            // The colors which are used for the picking scheme

var plotTranslations = [];            // Keep track of the translation for each plot.
var plotRotations = [];                // Keep track of the rotations for each plot
var zoomedInMatrix = -1;            // The matrix witch is currently zoomed in on
var clickedMatrix = -1;
var backupTranslations = [];
var backupRotations = [];
var animationStarted = false;
var alphaValueSpaceTime = 1.0;                // original alpha value for default plot        
var backupAlphaValue = alphaValueSpaceTime;    // backup used in animations
var minSelectedDelayValue = -1;
var maxSelectedDelayValue = -1;
var animationTimeout = 33; // 30Hz
var animationGranularity = 20;


function customMouseDown_SpaceTime(event) {
    if (!animationStarted) {
        if (clickedMatrix >= 0) {
            doZoomOutAnimation();
        } else {
            GL_handleMouseDown(event, '#' + CONNECTIVITY_SPACE_TIME_CANVAS_ID);
            doPick = true;
            drawSceneSpaceTime();
        }
    }
}

function initColorsForPicking() {
    colorsForPicking = [];
    for (var i=0; i <= nrOfSteps; i++) {
        // Go up to nrOfSteps since for 0 we will consider the full matrix as being clicked
        var r = parseInt(1.0 / (i + 1) * 255);
        var g = parseInt(i / nrOfSteps * 255);
        var b = 0.0;
        colorsForPicking.push([r / 255, g / 255, b / 255]);
        var colorKey = r + '' + g + '0';
        GL_colorPickerMappingDict[colorKey] = i;
    }
    GL_initColorPickFrameBuffer();
}

/*
 * Custom shader initializations specific for the space-time connectivity plot
 */
function initShaders_SPACETIME() {
    createAndUseShader("shader-plot-fs", "shader-plot-vs");
    SHADING_Context.basic_program_init(GL_shaderProgram);

    GL_shaderProgram.drawLines = gl.getUniformLocation(GL_shaderProgram, "uDrawLines");
    GL_shaderProgram.alphaValue = gl.getUniformLocation(GL_shaderProgram, "uAlpha");
    GL_shaderProgram.lineColor = gl.getUniformLocation(GL_shaderProgram, "uLineColor");
    GL_shaderProgram.isPicking = gl.getUniformLocation(GL_shaderProgram, "isPicking");
    GL_shaderProgram.pickingColor = gl.getUniformLocation(GL_shaderProgram, "pickingColor");
    
    GL_shaderProgram.vertexColorAttribute = gl.getAttribLocation(GL_shaderProgram, "aVertexColor");
    gl.enableVertexAttribArray(GL_shaderProgram.vertexColorAttribute);
}


function connectivitySpaceTime_startGL() {
    conectivitySpaceTime_initCanvas();
    //Do the required initializations for the connectivity space-time visualizer
    initShaders_SPACETIME();

    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    updateSpaceTimeHeader();
}

function _GL_createBuffer(data, type){
    type = type || gl.ARRAY_BUFFER;
    var buff = gl.createBuffer();
    gl.bindBuffer(type, buff);
    gl.bufferData(type, data, gl.STATIC_DRAW);
    buff.numItems = data.length;
    return buff;
}

/*
 * Create the required buffers for the space-time plot.
 */
function createConnectivityMatrix() {
    var nrElems = GVAR_interestAreaVariables[GVAR_selectedAreaType].values.length;
    // starting 'x' and 'y' axis values for the plot in order to center around (0, 0)
    var startX = - plotSize / 2;
    var startY = - plotSize / 2;
    // The size of a matrix element
    var elementSize = plotSize / nrElems;
    // Create arrays from start for performance reasons 
    var vertices = new Float32Array(nrElems * nrElems * 4 * 3);
    var normals = new Float32Array(nrElems * nrElems * 4 * 3);
    var indices = new Uint16Array(nrElems * nrElems * 2 * 3);
    var linesIndices = new Uint16Array(nrElems * nrElems * 2 * 4);

    var linearIndex = -1;

    for (var i = 0; i < nrElems; i++) {
        for (var j = 0; j < nrElems; j++) {
            linearIndex += 1;
            // For each separate element, compute the position of the 4 required vertices
            // depending on the position from the connectivity matrix
            var upperLeftX = startX + j * elementSize;
            var upperLeftY = startY + i * elementSize;

            // Since the vertex array is flatten, and there are 4 vertices per one element,
            // in order to fill the position in the vertice array we need to fill all 12 elements
            var elemVertices = [
                upperLeftX, upperLeftY, defaultZ,
                upperLeftX + elementSize, upperLeftY, defaultZ, //upper right
                upperLeftX, upperLeftY + elementSize, defaultZ, //lower left
                upperLeftX + elementSize, upperLeftY + elementSize, defaultZ // lower right
            ];

            var indexBase = 4 * linearIndex;

            // For the normals it's easier since we only need one normal for each vertex
            var elemNormals = [
                0, 0, -1,
                0, 0, -1,
                0, 0, -1,
                0, 0, -1
            ];

            // We have 2 triangles, which again are flatten so we need to fill 6 index elements
            var elemIndices = [
                indexBase + 0, indexBase + 1, indexBase + 2,
                indexBase + 1, indexBase + 2, indexBase + 3
            ];

            // For the lines we have 4 lines per element, flatten again, so 8 index elements to fill
            var elemLines = [
                indexBase + 0, indexBase + 1,
                indexBase + 1, indexBase + 3,
                indexBase + 2, indexBase + 3,
                indexBase + 2, indexBase + 0
            ];

            vertices.set(elemVertices, 3 * 4 * linearIndex);
            normals.set(elemNormals, 3 * 4 * linearIndex);
            indices.set(elemIndices, 3 * 2 * linearIndex);
            linesIndices.set(elemLines, 4 * 2 * linearIndex);
        }
    }
    // Now create all the required buffers having the computed data.
    verticesBuffer = _GL_createBuffer(vertices);
    normalsBuffer = _GL_createBuffer(normals);
    indexBuffer = _GL_createBuffer(indices, gl.ELEMENT_ARRAY_BUFFER);
    linesIndexBuffer = _GL_createBuffer(linesIndices, gl.ELEMENT_ARRAY_BUFFER);
    createOutlineSquare(startX, startY, elementSize, nrElems);
}

/*
 * Compute the required vertex and idex for the square outline of the full connectivity matrix
 */
function createOutlineSquare(startX, startY, elementSize, nrElems) {
    var width = nrElems * elementSize;
    var outlineVertices = [
        startX, startY, defaultZ,
        startX + width, startY, defaultZ,
        startX, startY + width, defaultZ,
        startX + width, startY + width, defaultZ
    ];
    var outlineNormals = [0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1];
    var outlineLines = [0, 1, 0, 2, 1, 3, 2, 3];
    outlineVerticeBuffer = _GL_createBuffer(new Float32Array(outlineVertices));
    outlineNormalsBuffer = _GL_createBuffer(new Float32Array(outlineNormals));
    outlineLinesBuffer = _GL_createBuffer(new Uint16Array(outlineLines), gl.ELEMENT_ARRAY_BUFFER);
}


/*
 * Generate a color buffer which represents the state of the weights for 
 * a given 'interval' centered around a given tract value
 * 
 * @param tractValue: the center of the interval
 * @param intervalLength: the length of the interval
 * 
 * @returns: a color buffer, where for the connections that fall in the defined interval,
 *              a gradient color is assigned based on the weights strenght, and for the 
 *              rest the color black is used.
 */
function generateColors(tractValue, intervalLength) {
    var theme = ColSchGetTheme().connectivityStepPlot;
    var matrixWeightsValues = GVAR_interestAreaVariables[1].values;
    var matrixTractsValues = GVAR_interestAreaVariables[2].values;
    var minWeightsValue = GVAR_interestAreaVariables[1].min_val;
    var maxWeightsValue = GVAR_interestAreaVariables[1].max_val;
    var nrElems = matrixWeightsValues.length;
    var colors = new Float32Array(nrElems * nrElems * 3 * 4);
    var linearIndex = -1;

    for (var i = 0; i < nrElems; i++) {
        for (var j = 0; j < nrElems; j++) {
            linearIndex += 1;
            // For each element generate 4 identical colors corresponding to the 4 vertices used for the element
            var delayValue = matrixTractsValues[i][nrElems - j - 1] / conductionSpeed;
            var weight = matrixWeightsValues[i][nrElems - j - 1];

            var isWithinInterval = (delayValue >= (tractValue - intervalLength / 2) &&
                                    delayValue <= (tractValue + intervalLength / 2));
            var color;

            if (isWithinInterval && weight != 0) {
                color = getGradientColor(weight, minWeightsValue, maxWeightsValue).slice(0, 3);
            }else{
                color = theme.noValueColor;
            }

            color = [].concat(color, color, color, color);
            colors.set(color, 3 * 4 * linearIndex);
        }
    }
    var buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    return buffer;
}


/**
 * Update the header with information about matrices.
 */ 
function updateSpaceTimeHeader() {

    function parseFloatDefault(valueUI, defaultValue){
        var resultValue = parseFloat(valueUI);
        if(isNaN(resultValue)){
            return defaultValue;
        }else{
            return resultValue;
        }
    }

    var fromDelaysInput = $('#fromDelaysValue');
    var toDelaysInput = $('#toDelaysValue');
    var conductionSpeedInput = $('#conductionSpeedValue');
    var currentMinDelay, currentMaxDelay;
    var uiConductionSpeed = parseFloatDefault(conductionSpeedInput.val(), conductionSpeed);

    // if conduction speed has changed adjust tract bounds
    if(uiConductionSpeed !== conductionSpeed){
        conductionSpeed = uiConductionSpeed;
        currentMinDelay = GVAR_interestAreaVariables[2].min_val / conductionSpeed;
        currentMaxDelay = GVAR_interestAreaVariables[2].max_val / conductionSpeed;
        _connectivitySpaceTimeUpdateLegend();

    } else {
        // read delay bounds specified by user in UI, in case invalid defaults are previous values
        currentMinDelay = parseFloatDefault(fromDelaysInput.val(), minSelectedDelayValue);
        currentMaxDelay = parseFloatDefault(toDelaysInput.val(), maxSelectedDelayValue);

        // ensure validity for tract bounds
        var maxAcceptedDelay = GVAR_interestAreaVariables[2].max_val / conductionSpeed;
        var minAcceptedDelay = GVAR_interestAreaVariables[2].min_val / conductionSpeed;

        if (currentMinDelay > currentMaxDelay) {
            var swapAux = currentMinDelay;
            currentMinDelay = currentMaxDelay;
            currentMaxDelay = swapAux;
        }
        if (currentMinDelay < minAcceptedDelay){
            currentMinDelay = minAcceptedDelay;
        }
        if (currentMaxDelay > maxAcceptedDelay || currentMaxDelay < 0) {
            currentMaxDelay = maxAcceptedDelay;
        }
    }

    //set globals
    minSelectedDelayValue = currentMinDelay;
    maxSelectedDelayValue = currentMaxDelay;

    //update ui
    fromDelaysInput.val(minSelectedDelayValue.toFixed(2));
    toDelaysInput.val(maxSelectedDelayValue.toFixed(2));
    conductionSpeedInput.val(conductionSpeed.toFixed(2));

    ConnStepPlotInitColorBuffers();
    if (clickedMatrix >= 0) {
        doZoomOutAnimation();
    } else {
        drawSceneSpaceTime();
    }
}


function ConnStepPlotInitColorBuffers() {
    initColorsForPicking();
    plotColorBuffers = [];
    var stepValue = (maxSelectedDelayValue - minSelectedDelayValue) / nrOfSteps;
    plotColorBuffers.push(generateColors((maxSelectedDelayValue + minSelectedDelayValue) / 2, maxSelectedDelayValue - minSelectedDelayValue));
    // In order to avoid floating number approximations which keep the loop for one more iteration just approximate by
    // substracting 0.1
    for (var tractValue = minSelectedDelayValue + stepValue / 2; tractValue < parseInt(maxSelectedDelayValue) - 0.1; tractValue = tractValue + stepValue) {
        plotColorBuffers.push(generateColors(tractValue, stepValue));
    } 
    var theme = ColSchGetTheme().connectivityStepPlot;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    drawSceneSpaceTime();
}

/*
 * Initialize the space time connectivity plot.
 */
function conectivitySpaceTime_initCanvas() {
    var canvas = document.getElementById(CONNECTIVITY_SPACE_TIME_CANVAS_ID);
    initGL(canvas);
    var theme = ColSchGetTheme().connectivityStepPlot;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    plotSize = parseInt(canvas.clientWidth / 3);    // Compute the size of one connectivity plot depending on the canvas width
    createConnectivityMatrix();
    canvas.onmousedown = customMouseDown_SpaceTime;

    plotTranslations = [];
    plotRotations = [];
    plotTranslations.push([-parseInt(canvas.clientWidth / 4), 0, 0]);    //The translation for the left-most full connectivity matrix
    plotRotations.push([90, [0, 1, 0]]);
    for (var i = 0; i < nrOfSteps; i++) {
        plotTranslations.push([-parseInt(canvas.clientWidth / 8) + parseInt(canvas.clientWidth / 2.2 / nrOfSteps) * i, 0.0, 0.0]); // Values tested that display nicely for 6 plots at least
        plotRotations.push([80 - i * nrOfSteps, [0, 1, 0]]);
    }
    
    if (minSelectedDelayValue < 0) {
        minSelectedDelayValue = GVAR_interestAreaVariables[2].min_val / conductionSpeed;
    }
    if (maxSelectedDelayValue < 0) {
        maxSelectedDelayValue = GVAR_interestAreaVariables[2].max_val / conductionSpeed;
    }
    
    clickedMatrix = -1;

    _connectivitySpaceTimeUpdateLegend();
}

function _connectivitySpaceTimeUpdateLegend(){
    $('#leg_min_tract').html(GVAR_interestAreaVariables[2].min_non_zero.toFixed(2));
    $('#leg_max_tract').html(GVAR_interestAreaVariables[2].max_val.toFixed(2));
    $('#leg_min_weights').html(GVAR_interestAreaVariables[1].min_non_zero.toFixed(2));
    $('#leg_max_weights').html(GVAR_interestAreaVariables[1].max_val.toFixed(2));
    $('#leg_conduction_speed').html(conductionSpeed.toFixed(2));
    $('#leg_min_delay').html((GVAR_interestAreaVariables[2].min_non_zero / conductionSpeed).toFixed(2));
    $('#leg_max_delay').html((GVAR_interestAreaVariables[2].max_val / conductionSpeed).toFixed(2));
}

/*
 * Draw the full matrix, with the outline square.
 */
function drawFullMatrix(doPick, idx) {
    var theme = ColSchGetTheme().connectivityStepPlot;
    mvPushMatrix();
    
    // Translate and rotate to get a good view 
    mvTranslate(plotTranslations[idx]);
    mvRotate(plotRotations[idx][0], plotRotations[idx][1]);
    mvRotate(180, [0, 0, 1]);
    
    // Draw the actual matrix.
    gl.bindBuffer(gl.ARRAY_BUFFER, verticesBuffer);
    gl.vertexAttribPointer(GL_shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);
//    gl.bindBuffer(gl.ARRAY_BUFFER, normalsBuffer);
//    gl.vertexAttribPointer(GL_shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);
    setMatrixUniforms();
    
    if (doPick) {
        var currentPickColor = colorsForPicking[idx];
        gl.uniform3f(GL_shaderProgram.pickingColor, currentPickColor[0], currentPickColor[1], currentPickColor[2]);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.drawElements(gl.TRIANGLES, indexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
    } else {
        gl.bindBuffer(gl.ARRAY_BUFFER, plotColorBuffers[idx]);
        gl.vertexAttribPointer(GL_shaderProgram.vertexColorAttribute, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.drawElements(gl.TRIANGLES, indexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
        gl.uniform3f(GL_shaderProgram.lineColor, theme.lineColor[0], theme.lineColor[1], theme.lineColor[2]);
        gl.uniform1i(GL_shaderProgram.drawLines, true);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, linesIndexBuffer);
        gl.lineWidth(1.0);
        gl.drawElements(gl.LINES, linesIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
        gl.uniform1i(GL_shaderProgram.drawLines, false);
        
        // Now draw the square outline
        if (idx == clickedMatrix) {
            gl.uniform3f(GL_shaderProgram.lineColor, theme.selectedOutlineColor[0], theme.selectedOutlineColor[1], theme.selectedOutlineColor[2]);
            gl.lineWidth(3.0);
        } else {
            gl.uniform3f(GL_shaderProgram.lineColor, theme.outlineColor[0], theme.outlineColor[1], theme.outlineColor[2]);
            gl.lineWidth(2.0);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, outlineVerticeBuffer);
        gl.vertexAttribPointer(GL_shaderProgram.vertexPositionAttribute, 3, gl.FLOAT, false, 0, 0);
//        gl.bindBuffer(gl.ARRAY_BUFFER, outlineNormalsBuffer);
//        gl.vertexAttribPointer(GL_shaderProgram.vertexNormalAttribute, 3, gl.FLOAT, false, 0, 0);
        gl.uniform1i(GL_shaderProgram.drawLines, true);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, outlineLinesBuffer);
        gl.drawElements(gl.LINES, outlineLinesBuffer.numItems, gl.UNSIGNED_SHORT, 0);
        gl.lineWidth(2.0);
        gl.uniform1i(GL_shaderProgram.drawLines, false);
    }

    mvPopMatrix();
}


function computeAnimationTranslations(startPoint, endPoint, startRotation, endRotation, startAlpha, endAlpha, animationSteps) {
    var translationAnimation = [];
    var rotationAnimation = [];
    var alphas = [];
    // y-axis component does not vary so we only handle x and z linearly
    var x_inc = (endPoint[0] - startPoint[0]) / animationSteps;
    var z_inc = (endPoint[2] - startPoint[2]) / animationSteps;
    // also we do rotations just along y-axis so angle of rotation is all we need to change
    var rot_inc = (endRotation[0] - startRotation[0]) / animationSteps;
    // for alpha alos vary linearly
    var alpha_inc = (endAlpha - startAlpha) / animationSteps;
    for (var i = 1; i <= animationSteps; i++) {
        translationAnimation.push([startPoint[0] + i * x_inc, startPoint[1], startPoint[2] + i * z_inc]);
        rotationAnimation.push([startRotation[0] + i * rot_inc, startRotation[1]]);
        alphas.push(startAlpha + alpha_inc * i);
    }
    return {'translations' : translationAnimation, 'rotations' : rotationAnimation, 'alphas' : alphas};
}


function animationStep(step, animationSteps, animations, zoomIn) {
    for (var j = 0; j < animations.length; j++) {
        plotTranslations[j] = animations[j]['translations'][step];
        plotRotations[j] = animations[j]['rotations'][step];
        alphaValueSpaceTime = animations[j]['alphas'][step];
    }
    drawSceneSpaceTime();
    if (step + 1 < animationSteps) {
        setTimeout(function() { animationStep(step + 1, animationSteps, animations, zoomIn); }, animationTimeout);
    } else {
        var matrixSelected = document.getElementById('selectedMatrixValue');
        if (zoomIn) {
            zoomedInMatrix = clickedMatrix;
            var stepValue = (maxSelectedDelayValue - minSelectedDelayValue) / nrOfSteps;
            if (zoomedInMatrix != 0) {
                var fromTractVal = (minSelectedDelayValue + stepValue * (zoomedInMatrix - 1)).toFixed(2);
                var toTractVal = (minSelectedDelayValue + stepValue * zoomedInMatrix).toFixed(2);
                matrixSelected.innerHTML = '[' + fromTractVal + '..' + toTractVal + ']';
            } else {
                matrixSelected.innerHTML = 'Full matrix';
            }
        } else {
            plotTranslations = backupTranslations;
            plotRotations = backupRotations;
            alphaValueSpaceTime = backupAlphaValue;
            clickedMatrix = -1;
            zoomedInMatrix = -1;
            matrixSelected.innerHTML = 'None';
        }
        drawSceneSpaceTime();
        animationStarted = false;
    }
}


function doZoomInAnimation() {
    animationStarted = true;
    var targetForwardPosition = [0.0, 0.0, 200];
    var targetForwardRotation = [360, [0, 1, 0]];
    backupTranslations = plotTranslations.slice(0);
    backupRotations  = plotRotations.slice(0);
    backupAlphaValue = alphaValueSpaceTime;
    var animations = [];
    for (var i = 0; i < plotTranslations.length; i++) {
        var targetTranslation, endRotation;
        if (i == clickedMatrix) {
            targetTranslation = targetForwardPosition;
            endRotation = targetForwardRotation;
        } else {
            targetTranslation = [plotTranslations[i][0], plotTranslations[i][1], -200];
            endRotation = plotRotations[i];
        }
        animations.push(computeAnimationTranslations(plotTranslations[i], targetTranslation,
                                                     plotRotations[i], endRotation,
                                                     alphaValueSpaceTime, 1, animationGranularity));
    }
    animationStep(0, animationGranularity, animations, true);
}


function doZoomOutAnimation() {
    animationStarted = true;
    var animations = [];
    for (var i = 0; i < plotTranslations.length; i++) {
        animations.push(computeAnimationTranslations(plotTranslations[i], backupTranslations[i],
                                                     plotRotations[i], backupRotations[i],
                                                     _alphaValue, backupAlphaValue, animationGranularity));
    }
    animationStep(0, animationGranularity, animations, false);
}


/*
 * Draw the entire space plot matrices.
 */
function drawSceneSpaceTime() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    // View angle is 45, we want to see object from 0.1 up to 800 distance from viewer
    perspective(45, gl.viewportWidth / gl.viewportHeight, near, 800.0);

    loadIdentity();
    // Translate to get a good view.
    mvTranslate([0.0, 0.0, -600]);

    if (!doPick) {
        gl.uniform1f(GL_shaderProgram.alphaValue, alphaValueSpaceTime);
        gl.uniform1f(GL_shaderProgram.isPicking, 0);
        gl.uniform3f(GL_shaderProgram.pickingColor, 1, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        for (var ii = 0; ii < plotColorBuffers.length; ii++) {
            drawFullMatrix(false, ii);
        }
    } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, GL_colorPickerBuffer);
           gl.disable(gl.BLEND);
        gl.disable(gl.DITHER);
           gl.uniform1f(GL_shaderProgram.isPicking, 1);
           gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        for (var i = 0; i < plotColorBuffers.length; i++) {
            drawFullMatrix(true, i);
        }
        clickedMatrix = GL_getPickedIndex();
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        doPick = false;
        if (clickedMatrix >= 0) {
            doZoomInAnimation();
        }
    }
    
}

