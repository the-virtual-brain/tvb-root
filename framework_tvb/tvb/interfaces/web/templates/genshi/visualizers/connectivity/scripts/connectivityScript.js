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

/* globals gl, GL_shaderProgram */

/**
 * WebGL methods "inheriting" from webGL_xx.js in static/js.
 */
var CONNECTIVITY_CANVAS_ID = "GLcanvas";

//NOTE: if <code>uColorIndex</code> has a value grater or equal to zero then the color corresponding to this index will
//      be used for drawing otherwise the color corresponding to the <code>aColorIndex</code> will be used for drawing


function initShaders() {	
    //INIT NORMAL SHADER
    createAndUseShader("shader-fs", "shader-vs");
    SHADING_Context.connectivity_init(GL_shaderProgram);
}

// if you change the no of colors (add/remove) please update also the size of the uniform 'uColorsArray' from 'x-vertex'
var COLORS = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 1.0],
        [0.1, 0.1, 0.1, 1.0]
];

var WHITE_COLOR_INDEX = 0;
var RED_COLOR_INDEX = 1;
var BLUE_COLOR_INDEX = 2;
var YELLOW_COLOR_INDEX = 3;
var GREEN_COLOR_INDEX = 4;
var GRAY_COLOR_INDEX = 5;
var BLACK_COLOR_INDEX = 6;

// the number of the points that were read from the 'position.txt' file (no of points from the connectivity matrix)
var NO_POSITIONS;
// each point from the connectivity matrix will be drawn as a square;
// each element of this array will contains: 1) the buffer with vertices needed for drawing the square;
// 2) the buffer with normals for each vertex of the square; 3) an array index buffer needed for drawing the square.
var positionsBuffers = [];
var positionsBuffers_3D = [];
// this list contains an array index buffer for each point from the connectivity matrix. The indices tell us between
// which points we should draw lines. All the lines that exit from a certain node.
var CONN_comingOutLinesIndices = [];
// this list contains an array index buffer for each point from the connectivity matrix. The indices tell us between
// which points we should draw lines. All the lines that enter in a certain node.
var CONN_comingInLinesIndices = [];
// represents a buffer which contains all the points from the connectivity matrix.
var positionsPointsBuffer;
// when we draw a line we have to specify the normals for the points between which the line is drawn;
// this buffer contains the normals (fake normals) for each point from the connectivity matrix.
var linesPointsNormalsBuffer;
// the index of the point that has to be highlight
var highlightedPointIndex1 = -1;
var highlightedPointIndex2 = -1;

// a buffer which contains for each point the index corresponding to the default color (white color)
var defaultColorIndexesBuffer;
// a buffer which contains for each point the index of a color that should be used for drawing it
var colorsArrayBuffer;
// this array contains a color index for each point from the connectivity matrix. The color corresponding to that index
// will be used for drawing the lines for that point
var colorsIndexes =[];
var conductionSpeed = 1;

var _alphaValue = 0.1;

var CONN_pickedIndex = -1;
var near = 0.1;
var doPick = false;

var showMetricDetails = false;

//when this var reaches to zero => all data needed for displaying the surface are loaded
var noOfBuffersToLoad = 3;

var colorsWeights = null;
var raysWeights = null;
var CONN_lineWidthsBins = [];

var maxLineWidth = 5.0;
var minLineWidth = 1.0;
var lineWidthNrOfBins = 10;

function toogleShowMetrics() {
    showMetricDetails = !showMetricDetails;
    drawScene();
}

function customKeyDown(event) {
    GL_handleKeyDown(event);
    GFUNC_updateLeftSideVisualization();
}

function customMouseDown(event) {
    GL_handleMouseDown(event, event.target);
    doPick = true;

    GFUNC_updateContextMenu(CONN_pickedIndex, GVAR_pointsLabels[CONN_pickedIndex],
           CONN_pickedIndex >= 0 && isAnyPointChecked(CONN_pickedIndex, CONN_comingInLinesIndices[CONN_pickedIndex], 0),
           CONN_pickedIndex >= 0 && isAnyPointChecked(CONN_pickedIndex, CONN_comingOutLinesIndices[CONN_pickedIndex], 1));
    GFUNC_updateLeftSideVisualization();
}

function customMouseMove(event) {
    GL_handleMouseMove(event);
    GFUNC_updateLeftSideVisualization();
}

function _customMouseWheelEvent(delta) {
    GL_handleMouseWeel(delta);
    GFUNC_updateLeftSideVisualization();
    return false; // prevent default
}


/**
 * Display the name for the selected connectivity node.
 */
function displayNameForPickedNode() {
    if (CONN_pickedIndex === undefined || CONN_pickedIndex < 0) {
        displayMessage("No node is currently highlighted.", "infoMessage");
    } else {
        displayMessage("The highlighted node is " + GVAR_pointsLabels[CONN_pickedIndex], "infoMessage");
    }
}

var linesBuffer;

function initBuffers() {
    var fakeNormal_1 = [0, 0, 0];

    var whitePointsColorsIndex =[];
    var points = [];
    var normals = [];

    for (var i = 0; i < NO_POSITIONS; i++) {
        points = points.concat(GVAR_positionsPoints[i]);
        normals = normals.concat(fakeNormal_1);
        colorsIndexes = colorsIndexes.concat(COLORS[WHITE_COLOR_INDEX]);       
    }

     for (var index = 0; index < 24 ; index++){
            whitePointsColorsIndex = whitePointsColorsIndex.concat(COLORS[WHITE_COLOR_INDEX]);
     }

    defaultColorIndexesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, defaultColorIndexesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(whitePointsColorsIndex), gl.STATIC_DRAW);
    defaultColorIndexesBuffer.itemSize = 3;
    defaultColorIndexesBuffer.numItems = parseInt(whitePointsColorsIndex.length);

    colorsArrayBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorsArrayBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colorsIndexes), gl.STATIC_DRAW);
    colorsArrayBuffer.itemSize = 3;
    colorsArrayBuffer.numItems = parseInt(colorsIndexes.length);

    positionsPointsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionsPointsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);
    positionsPointsBuffer.itemSize = 3;
    positionsPointsBuffer.numItems = parseInt(points.length / 3);

    linesPointsNormalsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, linesPointsNormalsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
    linesPointsNormalsBuffer.itemSize = 3;
    linesPointsNormalsBuffer.numItems = parseInt(normals.length / 3);

    createLinesBuffer([]);
}


function displayPoints() {
    gl.uniform1i(GL_shaderProgram.useVertexColors, false);
    for (var i = 0; i < NO_POSITIONS; i++) {
        // Next line was ADDED FOR PICK
        var currentBuffers;
        if (showMetricDetails) {
            currentBuffers = positionsBuffers_3D[i];
        } else {
            currentBuffers = positionsBuffers[i];
        }
        mvPickMatrix = GL_mvMatrix.dup();
        mvPushMatrix();

        if (colorsWeights) {
            // We have some color weights defined (eg. connectivity viewer)
            var color = getGradientColor(colorsWeights[i], parseFloat($('#colorMinId').val()), parseFloat($('#colorMaxId').val()));
            gl.uniform4f(GL_shaderProgram.materialColor, color[0], color[1], color[2], 1.0);
        }

        if (!showMetricDetails) {
            if (i == CONN_pickedIndex) {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[YELLOW_COLOR_INDEX]);
            } else if (i == highlightedPointIndex1) {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[RED_COLOR_INDEX]);
            } else if (i == highlightedPointIndex2) {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[BLUE_COLOR_INDEX]);
            } else if (GFUNC_isNodeAddedToInterestArea(i)) {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[GREEN_COLOR_INDEX]);
            } else if (GFUNC_isIndexInNodesWithPositiveWeight(i)) {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[BLUE_COLOR_INDEX]);
            } else if (!hasPositiveWeights(i)) {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[BLACK_COLOR_INDEX]);
            } else {
                gl.uniform4fv(GL_shaderProgram.materialColor, COLORS[WHITE_COLOR_INDEX]);
            }
        }
        // End ADDED FOR PICK
        setMatrixUniforms();
        SHADING_Context.connectivity_draw(GL_shaderProgram, currentBuffers[0], currentBuffers[1],
            currentBuffers[0], currentBuffers[2], gl.TRIANGLES);
        mvPopMatrix();
    }
    // Next line was ADDED FOR PICK
    doPick = false;
}


/**
 * Draw the light
 */
function addLight() {
    basicSetLighting();
    gl.uniform1f(GL_shaderProgram.alphaUniform, 1.0);
}

function addLightForCorticalSurface() {
    basicSetLighting();
    gl.uniform1f(GL_shaderProgram.alphaUniform, _alphaValue);
}

function drawScene() {
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    // View angle is 45, we want to see object from 0.1 up to 800 distance from viewer
    perspective(45, gl.viewportWidth / gl.viewportHeight, near, 800.0);

    mvPushMatrix();

    if (!doPick) {
        createLinesBuffer(getLinesIndexes()); // warn we upload new buffer each frame

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        addLight();

        //draw the lines between the checked points
        mvPushMatrix();
        mvTranslate([GVAR_additionalXTranslationStep, GVAR_additionalYTranslationStep, 0]);
        _drawLines(linesBuffer);
        mvPopMatrix();

        //draw the points
        mvPushMatrix();
        mvTranslate([GVAR_additionalXTranslationStep, GVAR_additionalYTranslationStep, 0]);
        displayPoints();
        mvPopMatrix();

       //ORIENTATION_draw_nose_and_ears();

        // draw the brain cortical surface
        if (noOfBuffersToLoad === 0) {
            mvPushMatrix();

            // Blending function for alpha: transparent pix blended over opaque -> opaque pix
            gl.enable(gl.BLEND);
            gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
            gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.enable(gl.CULL_FACE);
            addLightForCorticalSurface();
            mvTranslate([GVAR_additionalXTranslationStep, GVAR_additionalYTranslationStep, 0]);

            gl.uniform1i(GL_shaderProgram.useVertexColors, true);

            // Draw the transparent object twice, to get a correct rendering
            gl.cullFace(gl.FRONT);
            drawHemispheres(gl.TRIANGLES);
            gl.cullFace(gl.BACK);
            drawHemispheres(gl.TRIANGLES);

            gl.disable(gl.BLEND);
            gl.disable(gl.CULL_FACE);
            mvPopMatrix();
        }
    } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, GL_colorPickerBuffer);
        gl.disable(gl.BLEND);
        gl.disable(gl.DITHER);
        gl.uniform1f(GL_shaderProgram.useVertexColors, false);
        basicSetLighting(pickingLightSettings);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        if (GL_colorPickerInitColors.length == 0) {
            GL_initColorPickingData(NO_POSITIONS);
        }

        mvPushMatrix();
        mvTranslate([GVAR_additionalXTranslationStep, GVAR_additionalYTranslationStep, 0]);

        for (var i = 0; i < NO_POSITIONS; i++){
            gl.uniform4fv(GL_shaderProgram.materialColor, GL_colorPickerInitColors[i]);

            setMatrixUniforms();

            SHADING_Context.connectivity_draw(GL_shaderProgram, positionsBuffers[i][0], positionsBuffers[i][1],
                positionsBuffers[i][1], positionsBuffers[i][2], gl.TRIANGLES);
         }
        var newPicked = GL_getPickedIndex();
        if (newPicked != null) {
            CONN_pickedIndex = newPicked;
        }
        mvPopMatrix();
        doPick = false;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        drawScene();
    }
    mvPopMatrix();
}

/**
 * Given a list of indexes will create the buffer of elements needed to draw
 * line between the points that correspond to those indexes.
 */
function createLinesBuffer(dictOfIndexes) {
    linesBuffer = [];
    if (dictOfIndexes) {
        for (var width in dictOfIndexes) {
            var buffer = getElementArrayBuffer(dictOfIndexes[width]);
            buffer.lineWidth = width;
            linesBuffer.push(buffer);
        }
    }
}


function computeRay(rayWeight, minWeight, maxWeight) {
    var minRay = 1;
    var maxRay = 4;
    if (minWeight != maxWeight) {
        return minRay + [(rayWeight - minWeight) / (maxWeight - minWeight)] * (maxRay - minRay);
    }else{
        return minRay + (maxRay - minRay) / 2;
    }
}


/*
 * Get the aproximated line width value using a histogram like behaviour.
 */
function CONN_getLineWidthValue(weightsValue) {
    var minWeights = GVAR_interestAreaVariables[1].min_val;
    var maxWeights = GVAR_interestAreaVariables[1].max_val;
    if (maxWeights === minWeights) {
        maxWeights = minWeights + 1;
    }
    var lineDiff = maxLineWidth - minLineWidth;
    var scaling = (weightsValue - minWeights) / (maxWeights - minWeights);
    var lineWidth = scaling * lineDiff;
    var binInterval = lineDiff / lineWidthNrOfBins;
    return (parseInt(lineWidth / binInterval) * binInterval + minLineWidth).toFixed(6);
}


/*
 * Initialize the line widths histogram which can later on be used to plot lines
 * with different widths.
 */
function CONN_initLinesHistorgram() {
    CONN_lineWidthsBins = [];
    var weights = GVAR_interestAreaVariables[1].values;
    for (var i = 0; i < weights.length; i++) {
        var row = [];
        for (var j = 0; j < weights.length; j++) {
            row.push(CONN_getLineWidthValue(weights[i][j]));
        }
        CONN_lineWidthsBins.push(row);
    }
}


/**
 * Used for finding the indexes of the points that are connected by an edge. Will return a dictionary of the
 * form { line_width: [array of indices] } from which we can draw the lines using the array of indices with
 * using a given width.
 */
function getLinesIndexes() {
    var lines = {};
    var binInterval = (maxLineWidth - minLineWidth) / lineWidthNrOfBins;
    for (var bin = minLineWidth; bin <= maxLineWidth + 0.000001; bin += binInterval) {
        lines[bin.toFixed(6)] = [];
    }
    for (var i = 0; i < GVAR_connectivityMatrix.length; i++) {
        for (var j = 0; j < GVAR_connectivityMatrix[i].length; j++) {
            if (GVAR_connectivityMatrix[i][j] === 1) {
                try {
                    var bins = CONN_lineWidthsBins[i][j];
                    lines[bins].push(i);
                    lines[bins].push(j);
                } catch(err) {
                    console.log(err);
                }

            }
        }
    }
    return lines;
}

function _drawLines(linesBuffers) {
    gl.uniform1i(GL_shaderProgram.useVertexColors, true);

    setMatrixUniforms();
    
    for (var i = 0; i < linesBuffers.length; i++) {
        var linesVertexIndicesBuffer = linesBuffers[i];
        gl.lineWidth(parseFloat(linesVertexIndicesBuffer.lineWidth));

        SHADING_Context.connectivity_draw(GL_shaderProgram, positionsPointsBuffer, linesPointsNormalsBuffer,
            colorsArrayBuffer, linesVertexIndicesBuffer, gl.LINES);
    }
    gl.lineWidth(1.0);
}

/**
 * Create 2 dictionaries.
 * For each index keep a list of all incoming lines in one dictionary, and all outgoing lines in the other.
 */
function initLinesIndices() {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;
    for (var i = 0; i < NO_POSITIONS; i++) {
        var indexesIn = [];
        var indexesOut = [];
        for (var j = 0; j < NO_POSITIONS; j++) {
            if (j !== i && parseFloat(values[i][j])) {
                indexesOut.push(j);
            }
            if (j !== i && parseFloat(values[j][i])) {
                indexesIn.push(j);
            }
        }
        CONN_comingOutLinesIndices.push(indexesOut);
        CONN_comingInLinesIndices.push(indexesIn);
    }
}


function getElementArrayBuffer(indices) {
    var vertexIndices = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, vertexIndices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    vertexIndices.itemSize = 1;
    vertexIndices.numItems = indices.length;

    return vertexIndices;
}

/**
 * Method that handles the drawing/hiding of both coming in and coming out lines.
 * 
 * @param selectedNodeIdx the currently selected node
 * @param direction swap between outgoing(1) and ingoing(0)
 * @param draw swap between drawing(1) or hiding(0)
 */
function handleLines(selectedNodeIdx, direction, draw) {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;

    if (draw === 1)	{
        setCurrentColorForNode(selectedNodeIdx);
    }
    if (direction === 1) {
        //Coming out lines
        for (var i=0; i<NO_POSITIONS; i++) {
            if (values[selectedNodeIdx][i] > 0) {
                GVAR_connectivityMatrix[selectedNodeIdx][i] = draw;
            }
        }
    } else {
        for (var ii=0; ii<NO_POSITIONS; ii++) {
            if (values[ii][selectedNodeIdx] > 0) {
                GVAR_connectivityMatrix[ii][selectedNodeIdx] = draw;
            }
        }
    }
    drawScene();
}

/**
 * Draw all the comming in and comming out lines for the connectivity matrix.
 */
function checkAll() {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;
    for (var i = 0; i < NO_POSITIONS; i++) {
        var comingInEdgesIndexes = CONN_comingInLinesIndices[i];
        for (var j = 0; j < comingInEdgesIndexes.length; j++) {
            if (values[comingInEdgesIndexes[j]][i] > 0 ) {
                GVAR_connectivityMatrix[comingInEdgesIndexes[j]][i] = 1;
            }
        }
        var comingOutEdgesIndexes = CONN_comingOutLinesIndices[i];
        for (var jj = 0; jj < comingOutEdgesIndexes.length; jj++) {
            if (values[i][comingOutEdgesIndexes[jj]] > 0 ) {
                GVAR_connectivityMatrix[i][comingOutEdgesIndexes[jj]] = 1;
            }
        }
    }
    drawScene();
}

/**
 * Clear all the comming in and comming out lines for the connectivity matrix.
 */
function clearAll() {
    for (var i = 0; i < GVAR_connectivityMatrix.length; i++) {
        for (var j = 0; j < GVAR_connectivityMatrix[i].length; j++) {
            GVAR_connectivityMatrix[i][j] = 0;
        }
    }
    drawScene();
}


/**
 * Draw all the comming in and comming out lines for the connectivity matrix for selected nodes.
 */
function checkAllSelected() {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;

    for (var i = 0; i < NO_POSITIONS; i++) {
        if (GFUNC_isNodeAddedToInterestArea(i)) {
            var comingInEdgesIndexes = CONN_comingInLinesIndices[i];
            for (var j = 0; j < comingInEdgesIndexes.length; j++) {
                if (GFUNC_isNodeAddedToInterestArea(j) && values[comingInEdgesIndexes[j]][i] > 0) {
                    GVAR_connectivityMatrix[comingInEdgesIndexes[j]][i] = 1;
                }
            }
            var comingOutEdgesIndexes = CONN_comingOutLinesIndices[i];
            for (var jj = 0; jj < comingOutEdgesIndexes.length; jj++) {
                if (GFUNC_isNodeAddedToInterestArea(jj) && values[i][comingOutEdgesIndexes[jj]] > 0 ) {
                    GVAR_connectivityMatrix[i][comingOutEdgesIndexes[jj]] = 1;
                }
            }
        }
    }
    drawScene();
}

/**
 * Clear all the comming in and comming out lines for the connectivity matrix for selected nodes.
 */
function clearAllSelected() {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;

    for (var i = 0; i < NO_POSITIONS; i++) {
        if (GFUNC_isNodeAddedToInterestArea(i)) {
            var comingInEdgesIndexes = CONN_comingInLinesIndices[i];
            for (var j = 0; j < comingInEdgesIndexes.length; j++) {
                if (GFUNC_isNodeAddedToInterestArea(j) && values[comingInEdgesIndexes[j]][i] > 0) {
                    GVAR_connectivityMatrix[comingInEdgesIndexes[j]][i] = 0;
                }
            }
            var comingOutEdgesIndexes = CONN_comingOutLinesIndices[i];
            for (var jj = 0; jj < comingOutEdgesIndexes.length; jj++) {
                if (GFUNC_isNodeAddedToInterestArea(jj) && values[i][comingOutEdgesIndexes[jj]] > 0 ) {
                    GVAR_connectivityMatrix[i][comingOutEdgesIndexes[jj]] = 0;
                }
            }
      }
    }
    drawScene(); // todo: are 2 draw calls necessary?
    drawScene();
}


/**
 * Change the color that should be used for drawing the lines for the selected node
 *
 * @param selectedNodeIndex the index of the selected node
 */
function setCurrentColorForNode(selectedNodeIndex) {
    colorsIndexes[selectedNodeIndex*3] = getNewNodeColor()[0]/255.0;
    colorsIndexes[selectedNodeIndex*3+1] = getNewNodeColor()[1]/255.0;
    colorsIndexes[selectedNodeIndex*3+2] = getNewNodeColor()[2]/255.0;

    colorsArrayBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorsArrayBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colorsIndexes), gl.STATIC_DRAW);
    colorsArrayBuffer.itemSize = 3;
    colorsArrayBuffer.numItems = parseInt(colorsIndexes.length);
}


var verticesBuffers = [];
var normalsBuffers = [];
var indexesBuffers = [];
var hemispheres = ['leftHemisphere', 'rightLeftQuarter', 'leftRightQuarter', 'rightHemisphere'];
var GVAR_additionalXTranslationStep = 0;
var GVAR_additionalYTranslationStep = 0;


/**
 * Returns <code>true</code> if at least one point form the given list is checked.
 *
 * @param listOfIndexes the list of indexes for the points that should be verified. Each 2 elements from this list represent a point
 * (the indexes i and j from the connectivityMatrix in which is kept the information about the checked/unchecked points)
 * 
 * @param dir = 0 -> ingoing
 *        dir = 1 -> outgoing
 * 
 * @param idx -> point in question
 */
function isAnyPointChecked(idx, listOfIndexes, dir) {	
    for (var i = 0; i < listOfIndexes.length; i++) {
        var idx1 = listOfIndexes[i];
        if (dir === 0) {
            if (GVAR_connectivityMatrix[idx1][idx] === 1 ) {
                return true;
            }
        }
        if (dir === 1) {
            if (GVAR_connectivityMatrix[idx][idx1] === 1 ) {
                return true;
            }
        }
    }
    return false;
}

function hasPositiveWeights(i) {
    var values = GVAR_interestAreaVariables[GVAR_selectedAreaType].values;

    var hasWeights = false;
    for (var j = 0; j < NO_POSITIONS; j++) {
        if ((values[i][j] > 0 || values[j][i] > 0) && (i !== j)) {
            hasWeights = true;
        }
    }
    return hasWeights;
}
/**
 * Create webgl buffers from the specified files
 *
 * @param urlList the list of files urls
 * @param resultBuffers a list in which will be added the buffers created based on the data from the specified files
 * @param isIndex Boolean marking when current buffer to draw is with indexes.
 */
function getAsynchronousBuffers(urlList, resultBuffers, isIndex) {
    if (urlList.length === 0) {
        noOfBuffersToLoad -= 1;
        return;
    }
    $.get(urlList[0], function(data) {
        var dataList = eval(data);
        var buffer = gl.createBuffer();
        if (isIndex) {
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(dataList), gl.STATIC_DRAW);
            buffer.numItems = dataList.length;
        } else {
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(dataList), gl.STATIC_DRAW);
        }
        resultBuffers.push(buffer);
        urlList.splice(0, 1);
        return getAsynchronousBuffers(urlList, resultBuffers, isIndex);
    });
}


function selectHemisphere(index) {
    $(".quadrant-display").each(function () {
        $(this).removeClass('active');
    });
    $(".quadrant-"+ index).each(function () {
        $(this).addClass('active');
    });

    for (var k=0; k<hemispheres.length; k++){
        $("#" + hemispheres[k]).hide();
        $("#" + hemispheres[k]+'Tracts').hide();
    }
    $("#" + hemispheres[index]).show();
    $("#" + hemispheres[index]+'Tracts').show();
    var inputDiv = document.getElementById('editNodeValues');
    inputDiv.style.display = 'none';
}


/**
 * Method which draws the cortical surface
 */
function drawHemispheres(drawingMode) {
    gl.uniform1i(GL_shaderProgram.colorIndex, GRAY_COLOR_INDEX);
    for (var i = 0; i < verticesBuffers.length; i++) {
        //todo-io: hack for colors buffer
        //there should be passed an buffer of colors indexes not the normalBuffers;
        setMatrixUniforms();

        SHADING_Context.connectivity_draw(GL_shaderProgram, verticesBuffers[i], normalsBuffers[i],
            normalsBuffers[i], indexesBuffers[i], gl.TRIANGLES);
    }
}


/**
 * Contains GL initializations that need to be done each time the standard connectivity view is
 * selected from the available tabs.
 *
 * @param isSingleMode if is <code>true</code> that means that the connectivity will
 * be drawn alone, without widths and tracts.
 */
function connectivity_startGL(isSingleMode) {
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    if (!isSingleMode) {
        selectHemisphere(0);//mark is the gl init the right place for this??
    }
    GL_initColorPickFrameBuffer();
    drawScene();
}

function ConnPlotUpdateColors(){
    var theme = ColSchGetTheme().connectivityPlot;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    drawScene();
}
/**
 * Initialize the canvas and the event handlers. This should be called when switching from
 * other GL based visualizers to re-initiate the canvas.
 */
function connectivity_initCanvas() {
    var canvas = document.getElementById(CONNECTIVITY_CANVAS_ID);
    canvas.width = canvas.parentNode.clientWidth - 10;       // resize the canvas to almost fill the parent
    canvas.height = 550;
    initGL(canvas);
    initShaders();
    ColSch_initColorSchemeComponent();
    var theme = ColSchGetTheme().connectivityPlot;
    gl.clearColor(theme.backgroundColor[0], theme.backgroundColor[1], theme.backgroundColor[2], theme.backgroundColor[3]);
    // Enable keyboard and mouse interaction
    canvas.onkeydown = customKeyDown;
    canvas.onkeyup = GL_handleKeyUp;
    canvas.onmousedown = customMouseDown;
    $(document).on('mouseup', GL_handleMouseUp);
    $(document).on('mousemove', customMouseMove);

    patchContextMenu();
    $(canvas).contextMenu('#contextMenuDiv', {'appendTo': ".connectivity-viewer", 'shadow': false, 'offsetY': -13, 'offsetX': 0});
    $(canvas).click(displayNameForPickedNode);
    $(canvas).mousewheel(function(event, delta) { return _customMouseWheelEvent(delta); });
    canvas.redrawFunctionRef = drawScene;
}

/**
 * Initialize all the actual data needed by the connectivity visualizer. This should be called
 * only once.
 */
function saveRequiredInputs_con(fileWeights, fileTracts, filePositions, urlVerticesList, urlTrianglesList,
                                urlNormalsList, urlLabels, condSpeed, rays, colors) {
    GVAR_initPointsAndLabels(filePositions, urlLabels);
    NO_POSITIONS = GVAR_positionsPoints.length;
    GFUNC_initTractsAndWeights(fileWeights, fileTracts);
    if (rays) raysWeights = $.parseJSON(rays);
    if (colors) colorsWeights = $.parseJSON(colors);

    conductionSpeed = parseFloat(condSpeed);
    // Initialize the buffers for drawing the points
    var ray_value;

    for (var i = 0; i < NO_POSITIONS; i++) {
        if (raysWeights) {
            ray_value = computeRay(raysWeights[i], parseFloat($('#rayMinId').val()), parseFloat($('#rayMaxId').val()));
        }
        else {
            ray_value = 3;
        }
        positionsBuffers_3D[i] = HLPR_sphereBufferAtPoint(gl, GVAR_positionsPoints[i], ray_value);
        positionsBuffers[i] = HLPR_bufferAtPoint(gl, GVAR_positionsPoints[i]);
    }
    initBuffers();

    var urlVertices = $.parseJSON(urlVerticesList);
    if (urlVertices.length > 0) {
        var urlNormals = $.parseJSON(urlNormalsList);
        var urlTriangles = $.parseJSON(urlTrianglesList);
        getAsynchronousBuffers(urlVertices, verticesBuffers, false);
        getAsynchronousBuffers(urlNormals, normalsBuffers, false);
        getAsynchronousBuffers(urlTriangles, indexesBuffers, true);
    }
    GFUNC_initConnectivityMatrix(NO_POSITIONS);
    // Initialize the indices buffers for drawing the lines between the drawn points
    initLinesIndices();
    CONN_initLinesHistorgram();
}

/**
 * Change transparency of cortical surface from user-input.
 *
 * @param inputField user given input value for transparency of cortical-surface
 */
function changeSurfaceTransparency(inputField) {
    var newValue = inputField.value;

    if (!isNaN(parseFloat(newValue)) && isFinite(newValue) && parseFloat(newValue) >= 0 && parseFloat(newValue) <= 1) {
        _alphaValue = parseFloat(newValue);
    } else {
        inputField.value = _alphaValue;
        displayMessage("Transparency value should be a number between 0 and 1.", "warningMessage");
    }
}

/**
 * This will take all the required steps to start the connectivity visualizer.
 *
 * @param isSingleMode if is <code>true</code> that means that the connectivity will
 * be drawn alone, without widths and tracts.
 */
function prepareConnectivity(fileWeights, fileTracts, filePositions, urlVerticesList , urlTrianglesList,
                             urlNormalsList, urlLabels, isSingleMode, conductionSpeed, rays, colors) {
    connectivity_initCanvas();
    saveRequiredInputs_con(fileWeights, fileTracts, filePositions, urlVerticesList , urlTrianglesList,
                           urlNormalsList, urlLabels, conductionSpeed, rays, colors);
    ColSch_initColorSchemeComponent();
    connectivity_startGL(isSingleMode);
}

